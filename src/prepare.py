from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .columns import COLUMN_MAP, LIKERT_COLUMNS, RESOLUTION_MAP, TURNAROUND_MAP
from .utils import file_sha256, normalise_comment


def _validate_schema(raw: pd.DataFrame) -> None:
    missing = [label for label in COLUMN_MAP.values() if label not in raw.columns]
    if missing:
        raise ValueError("Workbook schema mismatch. Missing columns: " + "; ".join(missing))


def _map_categorical(series: pd.Series, mapping: dict[str, Any], name: str) -> pd.Series:
    observed = set(series.dropna().astype(str).unique())
    unknown = sorted(observed.difference(mapping))
    if unknown:
        raise ValueError(f"Unrecognised values in {name}: {unknown}")
    return series.map(mapping)


def load_and_prepare(
    path: str | Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    raw = pd.read_excel(input_path, sheet_name="Responses")
    _validate_schema(raw)
    raw_sha = file_sha256(input_path)

    data = raw.rename(columns={v: k for k, v in COLUMN_MAP.items()}).copy()
    data.insert(0, "source_row", np.arange(2, len(data) + 2))

    if pd.api.types.is_numeric_dtype(data["timestamp"]):
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce", origin="1899-12-30", unit="D"
        )
    else:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    numeric_columns = LIKERT_COLUMNS + [
        "overall_satisfaction",
        "reuse_intention",
        "recommendation_intention",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["turnaround_acceptability"] = _map_categorical(
        data["turnaround_acceptability"], TURNAROUND_MAP, "turnaround_acceptability"
    ).astype("Float64")
    data["resolution_status"] = _map_categorical(
        data["resolution_status"], RESOLUTION_MAP, "resolution_status"
    )

    required_ranges = {column: (1, 5) for column in LIKERT_COLUMNS}
    required_ranges.update(
        {
            "overall_satisfaction": (1, 10),
            "reuse_intention": (1, 10),
            "recommendation_intention": (1, 10),
        }
    )
    range_errors: dict[str, list[int]] = {}
    for column, (lower, upper) in required_ranges.items():
        invalid = data[column].notna() & ~data[column].between(lower, upper)
        if invalid.any():
            range_errors[column] = data.loc[invalid, "source_row"].astype(int).tolist()
    if range_errors:
        raise ValueError(f"Out-of-range values detected: {range_errors}")

    consent_ok = data["consent"].astype(str).str.strip().str.casefold().eq("yes")
    eligibility_ok = (
        data["eligible_recent_use"].astype(str).str.strip().str.casefold().eq("yes")
    )
    outcome_ok = data["overall_satisfaction"].notna()
    include = consent_ok & eligibility_ok & outcome_ok

    exclusion_reason = pd.Series("", index=data.index, dtype="object")
    exclusion_reason.loc[~consent_ok] = "Consent not confirmed"
    exclusion_reason.loc[consent_ok & ~eligibility_ok] = "Eligibility not confirmed"
    exclusion_reason.loc[consent_ok & eligibility_ok & ~outcome_ok] = "Missing overall satisfaction"
    exclusions = data.loc[~include, ["source_row"]].copy()
    exclusions["reason"] = exclusion_reason.loc[~include].values

    data = data.loc[include].reset_index(drop=True)
    data.insert(0, "response_id", np.arange(1, len(data) + 1))

    primary_max = int(config["primary_dissatisfaction_max_score"])
    sensitivity_max = int(config["sensitivity_dissatisfaction_max_score"])
    data["dissatisfied_primary"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data.loc[data["overall_satisfaction"].notna(), "dissatisfied_primary"] = (
        data.loc[data["overall_satisfaction"].notna(), "overall_satisfaction"] <= primary_max
    ).astype(int)
    data["dissatisfied_sensitivity"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    data.loc[data["overall_satisfaction"].notna(), "dissatisfied_sensitivity"] = (
        data.loc[data["overall_satisfaction"].notna(), "overall_satisfaction"] <= sensitivity_max
    ).astype(int)
    data["resolution_partial"] = data["resolution_status"].eq("Partly resolved").astype(int)
    data["resolution_unresolved"] = data["resolution_status"].eq("Not resolved").astype(int)

    exact_duplicates = int(raw.duplicated(keep=False).sum())
    structured_columns = [
        c
        for c in data.columns
        if c
        not in {
            "source_row",
            "response_id",
            "timestamp",
            "positive_comment",
            "negative_comment",
            "improvement_comment",
        }
    ]
    structured_duplicates = int(data.duplicated(subset=structured_columns, keep=False).sum())

    comment_duplicate_rows: dict[str, int] = {}
    comment_unique_counts: dict[str, int] = {}
    for column in ["positive_comment", "negative_comment", "improvement_comment"]:
        normalised = data[column].map(normalise_comment)
        present = normalised.ne("")
        comment_duplicate_rows[column] = int(
            normalised.loc[present].duplicated(keep=False).sum()
        )
        comment_unique_counts[column] = int(normalised.loc[present].nunique())

    model_eligible = data["resolution_status"].ne("Not sure")
    audit: dict[str, Any] = {
        "input_file": input_path.name,
        "input_sha256": raw_sha,
        "expected_input_sha256": config.get("expected_input_sha256"),
        "checksum_matches_expected": raw_sha == config.get("expected_input_sha256"),
        "submitted": int(len(raw)),
        "retained": int(len(data)),
        "excluded": int(len(exclusions)),
        "consented_submitted": int(consent_ok.sum()),
        "eligible_submitted": int(eligibility_ok.sum()),
        "exact_duplicate_rows": exact_duplicates,
        "structured_duplicate_rows_excluding_timestamp_and_comments": structured_duplicates,
        "missing_core_satisfaction": int(data["overall_satisfaction"].isna().sum()),
        "turnaround_applicable": int(data["turnaround_acceptability"].notna().sum()),
        "turnaround_structural_na": int(data["turnaround_acceptability"].isna().sum()),
        "resolution_not_sure": int(data["resolution_status"].eq("Not sure").sum()),
        "resolution_model_n": int(model_eligible.sum()),
        "primary_dissatisfied_all": int(data["dissatisfied_primary"].sum()),
        "primary_dissatisfied_model_eligible": int(
            data.loc[model_eligible, "dissatisfied_primary"].sum()
        ),
        "sensitivity_dissatisfied_all": int(data["dissatisfied_sensitivity"].sum()),
        "positive_comments": int(data["positive_comment"].notna().sum()),
        "negative_comments": int(data["negative_comment"].notna().sum()),
        "improvement_comments": int(data["improvement_comment"].notna().sum()),
        "unique_comment_counts": comment_unique_counts,
        "duplicate_comment_rows": comment_duplicate_rows,
    }
    return data, audit, exclusions
