from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_output_dirs, write_json


def build_output_index(output_dir: str | Path) -> pd.DataFrame:
    tables, figures = ensure_output_dirs(output_dir)
    rows = []
    for folder, kind in [(tables, "table"), (figures, "figure")]:
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.name != "output_index.csv":
                rows.append(
                    {
                        "type": kind,
                        "file": str(path.relative_to(Path(output_dir))),
                        "bytes": path.stat().st_size,
                    }
                )
    index = pd.DataFrame(rows)
    index.to_csv(tables / "output_index.csv", index=False)
    return index


def build_headline_results(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    tables = root / "tables"
    satisfaction = json.loads((tables / "satisfaction_summary.json").read_text())
    diagnostics = json.loads((tables / "ols_diagnostics.json").read_text())
    coefficients = pd.read_csv(tables / "ols_hc3_coefficients.csv")
    turnaround = pd.read_csv(tables / "turnaround_subset_model.csv")
    primary_summary = pd.read_csv(
        tables / "classification_primary_1_5_metric_summary.csv", header=[0, 1], index_col=[0, 1]
    )
    h4b = pd.read_csv(tables / "h4b_robust_binomial_odds_ratios.csv")
    themes = pd.read_csv(tables / "text_theme_counts_all_comments.csv")

    def coefficient(name: str) -> dict[str, float]:
        row = coefficients.loc[coefficients["predictor"].eq(name)].iloc[0]
        return {
            "B": float(row["B"]),
            "CI_low": float(row["CI_low"]),
            "CI_high": float(row["CI_high"]),
            "p": float(row["p"]),
            "partial_r_squared": float(row["partial_r_squared"]),
        }

    logistic = primary_summary.loc[("regularised_logistic", "fixed_0_50")]
    forest = primary_summary.loc[("random_forest", "fixed_0_50")]
    turnaround_row = turnaround.loc[turnaround["predictor"].eq("turnaround_acceptability")].iloc[0]
    headline = {
        "sample": {
            "n": satisfaction["n"],
            "mean_satisfaction": satisfaction["mean"],
            "primary_dissatisfied_n": satisfaction["primary_dissatisfied_n"],
            "primary_dissatisfied_pct": satisfaction["primary_dissatisfied_pct"],
        },
        "primary_ols": {
            "r_squared": diagnostics["r_squared"],
            "adjusted_r_squared": diagnostics["adjusted_r_squared"],
            "communication": coefficient("communication_quality"),
            "waiting": coefficient("waiting_acceptability"),
            "helpfulness": coefficient("staff_helpfulness"),
            "partial_resolution": coefficient("resolution_partial"),
            "unresolved": coefficient("resolution_unresolved"),
            "resolution_block_partial_r_squared": diagnostics[
                "resolution_block_partial_r_squared"
            ],
        },
        "turnaround_subset": {
            "B": float(turnaround_row["B"]),
            "CI_low": float(turnaround_row["CI_low"]),
            "CI_high": float(turnaround_row["CI_high"]),
            "p": float(turnaround_row["p"]),
        },
        "classification_primary_fixed_0_50": {
            "regularised_logistic": {
                metric: {
                    "mean": float(logistic[(metric, "mean")]),
                    "sd": float(logistic[(metric, "std")]),
                }
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
            },
            "random_forest": {
                metric: {
                    "mean": float(forest[(metric, "mean")]),
                    "sd": float(forest[(metric, "std")]),
                }
                for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
            },
        },
        "h4b": h4b.loc[
            h4b["predictor"].isin(["resolution_partial", "resolution_unresolved"]),
            ["predictor", "odds_ratio", "OR_CI_low", "OR_CI_high", "p"],
        ].to_dict(orient="records"),
        "reported_theme_counts": themes.loc[
            (
                themes["source_question"].isin(["negative_comment", "improvement_comment"])
                & themes["theme"].isin(
                    [
                        "communication_updates",
                        "explanation_clarity",
                        "process_ease_hand_offs",
                        "resolution_escalation",
                        "turnaround",
                    ]
                )
            )
        ].to_dict(orient="records"),
    }
    write_json(tables / "headline_results.json", headline)
    return headline
