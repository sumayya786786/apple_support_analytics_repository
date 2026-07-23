from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import comment_hash, ensure_output_dirs

THEMES = [
    "appointment_access",
    "waiting",
    "communication_updates",
    "staff_helpfulness",
    "technical_knowledge",
    "explanation_clarity",
    "options_cost_transparency",
    "resolution_escalation",
    "turnaround",
    "process_ease_hand_offs",
    "other",
]

COMMENT_COLUMNS = ["positive_comment", "negative_comment", "improvement_comment"]


def run_text_coding(
    data: pd.DataFrame,
    output_dir: str | Path,
    coding_map_path: str | Path,
) -> pd.DataFrame:
    tables, _ = ensure_output_dirs(output_dir)
    coding_map = pd.read_csv(coding_map_path, dtype={"comment_hash": str})
    required = {"source_question", "comment_hash", *THEMES}
    missing = sorted(required.difference(coding_map.columns))
    if missing:
        raise ValueError(f"Coding map is missing columns: {missing}")

    private_rows = []
    analysis_rows = []
    for source in COMMENT_COLUMNS:
        for _, row in data.loc[data[source].notna(), ["response_id", source]].iterrows():
            text = str(row[source])
            digest = comment_hash(source, text)
            private_rows.append(
                {
                    "response_id": int(row["response_id"]),
                    "source_question": source,
                    "comment": text,
                    "comment_hash": digest,
                }
            )
            analysis_rows.append(
                {
                    "response_id": int(row["response_id"]),
                    "source_question": source,
                    "comment_hash": digest,
                }
            )

    private_template = pd.DataFrame(private_rows)
    for theme in THEMES:
        private_template[theme] = ""
    private_template["coder_notes"] = ""
    private_template.to_csv(tables / "manual_text_coding_template_PRIVATE.csv", index=False)

    analysis = pd.DataFrame(analysis_rows).merge(
        coding_map, on=["source_question", "comment_hash"], how="left", validate="many_to_one"
    )
    if analysis[THEMES].isna().any().any():
        unmatched = analysis.loc[analysis[THEMES].isna().any(axis=1), ["source_question", "comment_hash"]]
        raise ValueError(
            "The coding map does not cover every comment. Unmatched hashes: "
            + unmatched.to_dict(orient="records").__repr__()
        )

    coded_response_level = analysis[["response_id", "source_question", "comment_hash", *THEMES]].copy()
    coded_response_level.to_csv(tables / "text_coding_response_level_hashes.csv", index=False)

    count_rows = []
    for source, group in coded_response_level.groupby("source_question"):
        for theme in THEMES:
            count_rows.append(
                {
                    "source_question": source,
                    "theme": theme,
                    "count": int(group[theme].sum()),
                    "valid_comments": int(len(group)),
                    "percent_of_valid_comments": float(100 * group[theme].mean()),
                }
            )
    counts = pd.DataFrame(count_rows)
    counts.to_csv(tables / "text_theme_counts_all_comments.csv", index=False)

    # Sensitivity analysis after collapsing repeated exact wording.
    unique = coded_response_level.drop_duplicates(subset=["source_question", "comment_hash"])
    unique_rows = []
    for source, group in unique.groupby("source_question"):
        for theme in THEMES:
            unique_rows.append(
                {
                    "source_question": source,
                    "theme": theme,
                    "unique_wordings_with_theme": int(group[theme].sum()),
                    "unique_wordings": int(len(group)),
                    "percent_of_unique_wordings": float(100 * group[theme].mean()),
                }
            )
    pd.DataFrame(unique_rows).to_csv(
        tables / "text_theme_counts_unique_wording_sensitivity.csv", index=False
    )
    return counts
