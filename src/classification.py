from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .columns import DISPLAY_LABELS, PRIMARY_PREDICTORS
from .utils import ensure_output_dirs, write_json

FEATURES = PRIMARY_PREDICTORS


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def _select_inner_f1_threshold(
    fitted_estimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: StratifiedKFold,
) -> float:
    probabilities = cross_val_predict(
        fitted_estimator,
        x_train,
        y_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), probabilities])
    scored = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        scored.append(
            (
                f1_score(y_train, predictions, zero_division=0),
                recall_score(y_train, predictions, zero_division=0),
                -abs(threshold - 0.5),
                float(threshold),
            )
        )
    return max(scored)[3]


def _calibration_statistics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    logit = np.log(clipped / (1 - clipped))
    fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
    return {
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "intercept_se": float(fit.bse[0]),
        "slope_se": float(fit.bse[1]),
    }


def _calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame({"observed": y, "probability": p})
    try:
        frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["probability"], bins=bins)
    return (
        frame.groupby("bin", observed=True)
        .agg(
            n=("observed", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_dissatisfied_proportion=("observed", "mean"),
        )
        .reset_index(drop=True)
    )


def _model_objects(config: dict[str, Any], seed: int):
    logistic_cfg = config["logistic"]
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=int(logistic_cfg["max_iter"]),
                    solver=str(logistic_cfg["solver"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic_grid = {
        "model__C": logistic_cfg["C_grid"],
        "model__class_weight": logistic_cfg["class_weight_grid"],
    }
    forest_cfg = config["random_forest"]
    forest = RandomForestClassifier(
        n_estimators=int(forest_cfg["n_estimators"]),
        max_depth=int(forest_cfg["max_depth"]),
        min_samples_leaf=int(forest_cfg["min_samples_leaf"]),
        class_weight=forest_cfg["class_weight"],
        random_state=seed,
    )
    return logistic, logistic_grid, forest


def _nested_evaluation(
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    target_name: str,
) -> dict[str, Any]:
    seed = int(config["random_seed"])
    outer = RepeatedStratifiedKFold(
        n_splits=int(config["outer_folds"]),
        n_repeats=int(config["outer_repeats"]),
        random_state=seed,
    )
    logistic, logistic_grid, forest = _model_objects(config, seed)

    fold_rows: list[dict[str, Any]] = []
    best_parameters: list[dict[str, Any]] = []
    probabilities: dict[str, dict[int, list[float]]] = {
        "regularised_logistic": defaultdict(list),
        "random_forest": defaultdict(list),
    }
    decisions_inner_f1: dict[int, list[int]] = defaultdict(list)
    threshold_rows: list[dict[str, Any]] = []
    importance_rows: list[np.ndarray] = []

    for fold, (train_index, test_index) in enumerate(outer.split(x, y), start=1):
        inner = StratifiedKFold(
            n_splits=int(config["inner_folds"]),
            shuffle=True,
            random_state=seed + fold,
        )
        search = GridSearchCV(
            logistic,
            logistic_grid,
            scoring=str(config["selection_metric"]),
            cv=inner,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x[train_index], y[train_index])
        logistic_probability = search.predict_proba(x[test_index])[:, 1]
        inner_threshold = _select_inner_f1_threshold(
            search.best_estimator_, x[train_index], y[train_index], inner
        )
        threshold_rows.append(
            {
                "target": target_name,
                "fold": fold,
                "inner_f1_threshold": inner_threshold,
                **search.best_params_,
            }
        )
        primary_metrics = _classification_metrics(
            y[test_index], logistic_probability, float(config["probability_threshold"])
        )
        sensitivity_metrics = _classification_metrics(
            y[test_index], logistic_probability, inner_threshold
        )
        fold_rows.append(
            {
                "target": target_name,
                "model": "regularised_logistic",
                "threshold_strategy": "fixed_0_50",
                "fold": fold,
                "n_test": len(test_index),
                "n_dissatisfied_test": int(y[test_index].sum()),
                **primary_metrics,
            }
        )
        fold_rows.append(
            {
                "target": target_name,
                "model": "regularised_logistic",
                "threshold_strategy": "inner_f1",
                "fold": fold,
                "n_test": len(test_index),
                "n_dissatisfied_test": int(y[test_index].sum()),
                **sensitivity_metrics,
            }
        )
        best_parameters.append({"target": target_name, "fold": fold, **search.best_params_})
        for index, probability in zip(test_index, logistic_probability):
            probabilities["regularised_logistic"][int(index)].append(float(probability))
        for index, decision in zip(test_index, logistic_probability >= inner_threshold):
            decisions_inner_f1[int(index)].append(int(decision))

        forest.fit(x[train_index], y[train_index])
        forest_probability = forest.predict_proba(x[test_index])[:, 1]
        fold_rows.append(
            {
                "target": target_name,
                "model": "random_forest",
                "threshold_strategy": "fixed_0_50",
                "fold": fold,
                "n_test": len(test_index),
                "n_dissatisfied_test": int(y[test_index].sum()),
                **_classification_metrics(
                    y[test_index], forest_probability, float(config["probability_threshold"])
                ),
            }
        )
        for index, probability in zip(test_index, forest_probability):
            probabilities["random_forest"][int(index)].append(float(probability))
        importance = permutation_importance(
            forest,
            x[test_index],
            y[test_index],
            scoring=str(config["selection_metric"]),
            n_repeats=int(config["random_forest"]["permutation_repeats"]),
            random_state=seed + fold,
        )
        importance_rows.append(importance.importances_mean)

    fold_table = pd.DataFrame(fold_rows)
    averaged_probabilities = {
        model: np.array([np.mean(values[index]) for index in range(len(y))])
        for model, values in probabilities.items()
    }
    inner_f1_decisions = np.array(
        [int(np.mean(decisions_inner_f1[index]) >= 0.5) for index in range(len(y))]
    )
    return {
        "fold_table": fold_table,
        "best_parameters": pd.DataFrame(best_parameters),
        "threshold_table": pd.DataFrame(threshold_rows),
        "averaged_probabilities": averaged_probabilities,
        "inner_f1_decisions": inner_f1_decisions,
        "permutation_importance": np.vstack(importance_rows),
    }


def _write_model_outputs(
    result: dict[str, Any],
    y: np.ndarray,
    target_name: str,
    tables: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_table = result["fold_table"]
    fold_table.to_csv(tables / f"classification_{target_name}_fold_metrics.csv", index=False)
    summary = (
        fold_table.groupby(["model", "threshold_strategy"])[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
        ]
        .agg(["mean", "std"])
    )
    summary.to_csv(tables / f"classification_{target_name}_metric_summary.csv")
    result["best_parameters"].to_csv(
        tables / f"classification_{target_name}_logistic_best_parameters.csv", index=False
    )
    result["threshold_table"].to_csv(
        tables / f"classification_{target_name}_inner_thresholds.csv", index=False
    )

    aggregate: dict[str, Any] = {}
    for model, probabilities in result["averaged_probabilities"].items():
        predictions = (probabilities >= float(config["probability_threshold"])).astype(int)
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        pd.DataFrame(
            matrix,
            index=["actual_not_dissatisfied", "actual_dissatisfied"],
            columns=["pred_not_dissatisfied", "pred_dissatisfied"],
        ).to_csv(tables / f"classification_{target_name}_{model}_confusion_matrix.csv")
        predictions_table = pd.DataFrame(
            {
                "analysis_row": np.arange(1, len(y) + 1),
                "observed": y,
                "probability": probabilities,
                "predicted_fixed_0_50": predictions,
            }
        )
        if model == "regularised_logistic":
            predictions_table["predicted_inner_f1_majority"] = result["inner_f1_decisions"]
        predictions_table.to_csv(
            tables / f"classification_{target_name}_{model}_averaged_oof_predictions.csv",
            index=False,
        )
        calibration = _calibration_statistics(y, probabilities)
        write_json(
            tables / f"classification_{target_name}_{model}_calibration.json", calibration
        )
        _calibration_bins(y, probabilities).to_csv(
            tables / f"classification_{target_name}_{model}_calibration_bins.csv",
            index=False,
        )
        aggregate[model] = {
            "confusion_matrix": matrix.tolist(),
            "averaged_oof_metrics_fixed_0_50": _classification_metrics(
                y, probabilities, float(config["probability_threshold"])
            ),
            "calibration": calibration,
        }

    importance = result["permutation_importance"]
    pd.DataFrame(
        {
            "feature": FEATURES,
            "label": [DISPLAY_LABELS[feature] for feature in FEATURES],
            "mean_validation_permutation_importance": importance.mean(axis=0),
            "sd_validation_permutation_importance": importance.std(axis=0, ddof=1),
        }
    ).sort_values("mean_validation_permutation_importance", ascending=False).to_csv(
        tables / f"classification_{target_name}_random_forest_permutation_importance.csv",
        index=False,
    )
    return aggregate


def _h4b_robust_binomial(data: pd.DataFrame, tables: Path) -> pd.DataFrame:
    frame = data.loc[data["resolution_status"].ne("Not sure")].dropna(
        subset=FEATURES + ["dissatisfied_primary"]
    )
    x = sm.add_constant(frame[FEATURES].astype(float), has_constant="add")
    y = frame["dissatisfied_primary"].astype(float)
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(cov_type="HC3")
    rows = []
    for name in x.columns:
        coefficient = float(fit.params[name])
        se = float(fit.bse[name])
        rows.append(
            {
                "predictor": name,
                "coefficient_log_odds": coefficient,
                "SE_HC3": se,
                "odds_ratio": float(np.exp(coefficient)),
                "OR_CI_low": float(np.exp(coefficient - 1.959963984540054 * se)),
                "OR_CI_high": float(np.exp(coefficient + 1.959963984540054 * se)),
                "p": float(fit.pvalues[name]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(tables / "h4b_robust_binomial_odds_ratios.csv", index=False)
    return table


def run_classification(
    data: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    tables, figures = ensure_output_dirs(output_dir)
    frame = data.loc[data["resolution_status"].ne("Not sure")].dropna(subset=FEATURES).copy()
    x = frame[FEATURES].to_numpy(dtype=float)

    target_results = {}
    for target_name, target_column in [
        ("primary_1_5", "dissatisfied_primary"),
        ("sensitivity_1_6", "dissatisfied_sensitivity"),
    ]:
        y = frame[target_column].to_numpy(dtype=int)
        result = _nested_evaluation(x, y, config, target_name)
        aggregate = _write_model_outputs(result, y, target_name, tables, config)
        target_results[target_name] = {
            "result": result,
            "aggregate": aggregate,
            "y": y,
        }

    # Majority-class baseline for the primary outcome.
    y_primary = target_results["primary_1_5"]["y"]
    majority_prediction = np.zeros_like(y_primary)
    baseline = {
        "accuracy": float(accuracy_score(y_primary, majority_prediction)),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "pr_auc": float(y_primary.mean()),
        "brier": float(brier_score_loss(y_primary, np.repeat(y_primary.mean(), len(y_primary)))),
    }
    write_json(tables / "classification_majority_baseline.json", baseline)

    h4b = _h4b_robust_binomial(data, tables)

    primary_fold = target_results["primary_1_5"]["result"]["fold_table"]
    plot_summary = (
        primary_fold.loc[primary_fold["threshold_strategy"].eq("fixed_0_50")]
        .groupby("model")[["accuracy", "precision", "recall", "f1"]]
        .mean()
    )
    plot_summary.loc["majority_baseline"] = [baseline[key] for key in ["accuracy", "precision", "recall", "f1"]]
    plot_summary.to_csv(tables / "classification_primary_plot_summary.csv")

    # Figure 6: threshold performance comparison.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metrics = ["accuracy", "precision", "recall", "f1"]
    positions = np.arange(len(metrics))
    width = 0.25
    model_order = ["majority_baseline", "regularised_logistic", "random_forest"]
    for offset, model in enumerate(model_order):
        ax.bar(
            positions + (offset - 1) * width,
            plot_summary.loc[model, metrics],
            width,
            label=model.replace("_", " ").title(),
        )
    ax.set_xticks(positions, [metric.title() for metric in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Out-of-fold classification performance at threshold 0.50")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure_6_classification_performance.png", dpi=300)
    plt.close(fig)

    # Figure 7: calibration curves.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration")
    for model in ["regularised_logistic", "random_forest"]:
        probabilities = target_results["primary_1_5"]["result"]["averaged_probabilities"][model]
        bins = _calibration_bins(y_primary, probabilities)
        ax.plot(
            bins["mean_predicted_probability"],
            bins["observed_dissatisfied_proportion"],
            marker="o",
            label=model.replace("_", " ").title(),
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed dissatisfied proportion",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Out-of-fold calibration",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure_7_classification_calibration.png", dpi=300)
    plt.close(fig)

    summary = {
        "model_n": int(len(frame)),
        "primary_events": int(y_primary.sum()),
        "sensitivity_events": int(target_results["sensitivity_1_6"]["y"].sum()),
        "baseline": baseline,
        "h4b_resolution": h4b.loc[
            h4b["predictor"].isin(["resolution_partial", "resolution_unresolved"])
        ].to_dict(orient="records"),
    }
    write_json(tables / "classification_summary.json", summary)
    return summary
