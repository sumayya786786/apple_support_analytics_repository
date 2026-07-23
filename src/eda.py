from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .columns import DISPLAY_LABELS, RATING_COLUMNS
from .utils import ensure_output_dirs, write_json


def _cronbach_alpha_two_items(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a, b], axis=1).dropna().astype(float)
    item_variances = frame.var(ddof=1).sum()
    total_variance = frame.sum(axis=1).var(ddof=1)
    return float(2 / 1 * (1 - item_variances / total_variance))


def _frequency_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = data[column].value_counts(dropna=False)
    return pd.DataFrame(
        {
            "category": counts.index.astype(str),
            "n": counts.values,
            "percent": 100 * counts.values / len(data),
        }
    )


def _kruskal_by_group(data: pd.DataFrame, group: str) -> dict[str, Any]:
    valid = data.dropna(subset=[group, "overall_satisfaction"])
    samples = [g["overall_satisfaction"].astype(float).to_numpy() for _, g in valid.groupby(group)]
    if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
        return {"group": group, "n_groups": len(samples), "H": None, "p": None}
    result = stats.kruskal(*samples)
    return {
        "group": group,
        "n_groups": len(samples),
        "H": float(result.statistic),
        "p": float(result.pvalue),
    }


def run_eda(data: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    tables, figures = ensure_output_dirs(output_dir)

    satisfaction = data["overall_satisfaction"].dropna().astype(float)
    satisfaction_summary = {
        "n": int(len(satisfaction)),
        "mean": float(satisfaction.mean()),
        "sd": float(satisfaction.std(ddof=1)),
        "median": float(satisfaction.median()),
        "q1": float(satisfaction.quantile(0.25)),
        "q3": float(satisfaction.quantile(0.75)),
        "minimum": float(satisfaction.min()),
        "maximum": float(satisfaction.max()),
        "primary_dissatisfied_n": int(data["dissatisfied_primary"].sum()),
        "primary_dissatisfied_pct": float(100 * data["dissatisfied_primary"].mean()),
        "sensitivity_dissatisfied_n": int(data["dissatisfied_sensitivity"].sum()),
        "sensitivity_dissatisfied_pct": float(100 * data["dissatisfied_sensitivity"].mean()),
        "spearman_reuse": float(
            data[["overall_satisfaction", "reuse_intention"]].corr(method="spearman").iloc[0, 1]
        ),
        "spearman_recommendation": float(
            data[["overall_satisfaction", "recommendation_intention"]]
            .corr(method="spearman")
            .iloc[0, 1]
        ),
    }
    write_json(tables / "satisfaction_summary.json", satisfaction_summary)
    data["overall_satisfaction"].value_counts().sort_index().rename_axis("score").reset_index(name="n").to_csv(
        tables / "satisfaction_frequencies.csv", index=False
    )

    attribute_rows = []
    for column in RATING_COLUMNS:
        series = data[column].dropna().astype(float)
        attribute_rows.append(
            {
                "attribute": column,
                "label": DISPLAY_LABELS[column],
                "n": int(len(series)),
                "mean": float(series.mean()),
                "sd": float(series.std(ddof=1)),
                "rated_1_2_n": int(series.le(2).sum()),
                "rated_1_2_pct": float(100 * series.le(2).mean()),
                "neutral_n": int(series.eq(3).sum()),
                "favourable_4_5_n": int(series.ge(4).sum()),
                "spearman_with_satisfaction": float(
                    series.corr(data.loc[series.index, "overall_satisfaction"], method="spearman")
                ),
            }
        )
    attribute_summary = pd.DataFrame(attribute_rows)
    attribute_summary.to_csv(tables / "attribute_summary.csv", index=False)

    correlation = data[RATING_COLUMNS].corr(method="spearman")
    correlation.to_csv(tables / "predictor_spearman_matrix.csv")
    transparency = {
        "spearman_q14_q15": float(
            data[["options_clarity", "cost_warranty_clarity"]].corr(method="spearman").iloc[0, 1]
        ),
        "cronbach_alpha_two_item": _cronbach_alpha_two_items(
            data["options_clarity"], data["cost_warranty_clarity"]
        ),
        "decision": "Retain Q14 and Q15 separately",
    }
    write_json(tables / "transparency_reliability.json", transparency)

    resolution_data = data.loc[
        data["resolution_status"].isin(["Fully resolved", "Partly resolved", "Not resolved"]),
        ["resolution_status", "overall_satisfaction"],
    ].dropna()
    group_order = ["Fully resolved", "Partly resolved", "Not resolved"]
    resolution_summary = (
        resolution_data.groupby("resolution_status")["overall_satisfaction"]
        .agg(n="count", mean="mean", sd="std", median="median")
        .reindex(group_order)
        .reset_index()
    )
    resolution_summary.to_csv(tables / "resolution_satisfaction_summary.csv", index=False)
    groups = [
        resolution_data.loc[resolution_data["resolution_status"].eq(group), "overall_satisfaction"].to_numpy()
        for group in group_order
    ]
    kw = stats.kruskal(*groups)
    epsilon_squared = (kw.statistic - len(groups) + 1) / (len(resolution_data) - len(groups))
    pair_rows = []
    raw_p = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for left, right in pairs:
        result = stats.mannwhitneyu(groups[left], groups[right], alternative="two-sided", method="auto")
        raw_p.append(result.pvalue)
        pair_rows.append(
            {
                "group_1": group_order[left],
                "group_2": group_order[right],
                "U": float(result.statistic),
                "p_raw": float(result.pvalue),
            }
        )
    adjusted = multipletests(raw_p, method="holm")[1]
    for row, p_adjusted in zip(pair_rows, adjusted):
        row["p_holm"] = float(p_adjusted)
    pd.DataFrame(pair_rows).to_csv(tables / "resolution_pairwise_holm.csv", index=False)
    resolution_test = {
        "n": int(len(resolution_data)),
        "H": float(kw.statistic),
        "degrees_of_freedom": len(groups) - 1,
        "p": float(kw.pvalue),
        "epsilon_squared": float(epsilon_squared),
    }
    write_json(tables / "resolution_kruskal_wallis.json", resolution_test)

    context_rows = [
        _kruskal_by_group(data, group)
        for group in ["support_type", "warranty_status", "visit_recency", "issue_type"]
    ]
    pd.DataFrame(context_rows).to_csv(tables / "context_kruskal_wallis.csv", index=False)

    profile_columns = [
        "age_group",
        "gender",
        "employment_status",
        "support_type",
        "product_type",
        "issue_type",
        "visit_recency",
        "warranty_status",
        "resolution_status",
        "biggest_factor",
    ]
    profile_frames = []
    for column in profile_columns:
        table = _frequency_table(data, column)
        table.insert(0, "variable", column)
        profile_frames.append(table)
    pd.concat(profile_frames, ignore_index=True).to_csv(
        tables / "sample_profile_tables.csv", index=False
    )

    biggest = _frequency_table(data, "biggest_factor")
    biggest.to_csv(tables / "biggest_factor_counts.csv", index=False)

    # Figure 2: overall satisfaction distribution.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = data["overall_satisfaction"].value_counts().sort_index()
    x = np.arange(1, 11)
    y = counts.reindex(x, fill_value=0).to_numpy()
    ax.bar(x, y)
    ax.axvline(5.5, linestyle="--", linewidth=1.2, label="Primary dissatisfaction cut-off")
    ax.set(
        xlabel="Overall satisfaction score",
        ylabel="Number of responses",
        title=f"Distribution of overall satisfaction (n={len(data)})",
        xticks=x,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure_2_satisfaction_distribution.png", dpi=300)
    plt.close(fig)

    # Figure 3: diverging service-quality profile.
    plot_rows = []
    for column in RATING_COLUMNS:
        s = data[column].dropna().astype(float)
        plot_rows.append(
            {
                "label": DISPLAY_LABELS[column],
                "unfavourable": 100 * s.le(2).mean(),
                "neutral": 100 * s.eq(3).mean(),
                "favourable": 100 * s.ge(4).mean(),
            }
        )
    profile = pd.DataFrame(plot_rows)
    profile.to_csv(tables / "likert_profile_percentages.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(profile))
    ax.barh(positions, -profile["unfavourable"], label="Unfavourable (1–2)")
    ax.barh(positions, profile["neutral"], label="Neutral (3)")
    ax.barh(
        positions,
        profile["favourable"],
        left=profile["neutral"],
        label="Favourable (4–5)",
    )
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(positions, profile["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of valid responses")
    ax.set_title("Service-quality rating profile")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "figure_3_service_quality_profile.png", dpi=300)
    plt.close(fig)

    # Figure 4: Spearman heatmap.
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    labels = [DISPLAY_LABELS[column] for column in RATING_COLUMNS]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Spearman correlations among service-quality predictors")
    fig.tight_layout()
    fig.savefig(figures / "figure_4_predictor_correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # Figure 8: respondent-selected biggest factor.
    factor_plot = biggest.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(factor_plot["category"], factor_plot["n"])
    ax.set(
        xlabel="Number of respondents",
        title="Respondent-selected factor with the greatest reported impact",
    )
    fig.tight_layout()
    fig.savefig(figures / "figure_8_biggest_factor.png", dpi=300)
    plt.close(fig)

    return {
        "satisfaction": satisfaction_summary,
        "transparency": transparency,
        "resolution_test": resolution_test,
    }
