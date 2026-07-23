from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .columns import DISPLAY_LABELS, EXPLORATORY_PREDICTORS, PRIMARY_PREDICTORS
from .utils import ensure_output_dirs, write_json


def _primary_data(data: pd.DataFrame) -> pd.DataFrame:
    required = ["overall_satisfaction"] + PRIMARY_PREDICTORS
    return (
        data.loc[data["resolution_status"].ne("Not sure")]
        .dropna(subset=required)
        .copy()
    )


def _fit_ols_hc3(frame: pd.DataFrame, predictors: list[str]):
    x = sm.add_constant(frame[predictors].astype(float), has_constant="add")
    y = frame["overall_satisfaction"].astype(float)
    conventional = sm.OLS(y, x).fit()
    robust = conventional.get_robustcov_results(cov_type="HC3", use_t=False)
    return x, y, conventional, robust


def _partial_r2(full_frame: pd.DataFrame, predictors: list[str], full_r2: float) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        reduced = [item for item in predictors if item != predictor]
        reduced_fit = sm.OLS(
            full_frame["overall_satisfaction"].astype(float),
            sm.add_constant(full_frame[reduced].astype(float), has_constant="add"),
        ).fit()
        increment = full_r2 - reduced_fit.rsquared
        partial = increment / (1 - reduced_fit.rsquared)
        rows.append(
            {
                "predictor": predictor,
                "reduced_r_squared": float(reduced_fit.rsquared),
                "incremental_r_squared": float(increment),
                "partial_r_squared": float(partial),
            }
        )
    return pd.DataFrame(rows)


def run_explanatory(
    data: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    tables, figures = ensure_output_dirs(output_dir)
    model_data = _primary_data(data)
    x, y, conventional, robust = _fit_ols_hc3(model_data, PRIMARY_PREDICTORS)
    ci = robust.conf_int(alpha=0.05)

    y_sd = float(y.std(ddof=1))
    coefficient_rows = []
    for index, name in enumerate(x.columns):
        standardised_beta = np.nan
        y_standardised_discrete_effect = np.nan
        if name in {"waiting_acceptability", "communication_quality", "staff_helpfulness"}:
            standardised_beta = float(
                robust.params[index] * model_data[name].std(ddof=1) / y_sd
            )
        elif name in {"resolution_partial", "resolution_unresolved"}:
            y_standardised_discrete_effect = float(robust.params[index] / y_sd)
        coefficient_rows.append(
            {
                "predictor": name,
                "label": "Intercept" if name == "const" else DISPLAY_LABELS[name],
                "B": float(robust.params[index]),
                "SE_HC3": float(robust.bse[index]),
                "CI_low": float(ci[index, 0]),
                "CI_high": float(ci[index, 1]),
                "p": float(robust.pvalues[index]),
                "standardised_beta_continuous": standardised_beta,
                "y_standardised_discrete_effect": y_standardised_discrete_effect,
            }
        )
    coefficient_table = pd.DataFrame(coefficient_rows)

    importance = _partial_r2(model_data, PRIMARY_PREDICTORS, conventional.rsquared)
    coefficient_table = coefficient_table.merge(importance, on="predictor", how="left")
    coefficient_table.to_csv(tables / "ols_hc3_coefficients.csv", index=False)

    reduced_resolution_fit = sm.OLS(
        y,
        sm.add_constant(
            model_data[["waiting_acceptability", "communication_quality", "staff_helpfulness"]].astype(float),
            has_constant="add",
        ),
    ).fit()
    resolution_increment = conventional.rsquared - reduced_resolution_fit.rsquared
    resolution_partial_r2 = resolution_increment / (1 - reduced_resolution_fit.rsquared)

    residuals = conventional.resid
    bp = het_breuschpagan(residuals, x)
    white = het_white(residuals, x)
    jb = stats.jarque_bera(residuals)
    influence = conventional.get_influence()
    cooks = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    studentised = influence.resid_studentized_external
    influence_table = pd.DataFrame(
        {
            "response_id": model_data["response_id"].astype(int).to_numpy(),
            "fitted": conventional.fittedvalues,
            "residual": residuals,
            "studentised_residual": studentised,
            "leverage": leverage,
            "cooks_distance": cooks,
            "above_4_over_n": cooks > 4 / len(model_data),
        }
    ).sort_values("cooks_distance", ascending=False)
    influence_table.to_csv(tables / "ols_influence_observations.csv", index=False)

    vif_rows = []
    for idx, column in enumerate(x.columns):
        if column == "const":
            continue
        vif_rows.append(
            {"predictor": column, "VIF": float(variance_inflation_factor(x.values, idx))}
        )
    pd.DataFrame(vif_rows).to_csv(tables / "ols_vif.csv", index=False)

    diagnostics = {
        "n": int(conventional.nobs),
        "r_squared": float(conventional.rsquared),
        "adjusted_r_squared": float(conventional.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "model_F": float(conventional.fvalue),
        "model_F_df_model": int(conventional.df_model),
        "model_F_df_resid": int(conventional.df_resid),
        "model_F_p": float(conventional.f_pvalue),
        "breusch_pagan_LM": float(bp[0]),
        "breusch_pagan_p": float(bp[1]),
        "white_LM": float(white[0]),
        "white_p": float(white[1]),
        "jarque_bera_statistic": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "durbin_watson_descriptive_only": float(sm.stats.stattools.durbin_watson(residuals)),
        "max_cooks_distance": float(cooks.max()),
        "max_cooks_response_id": int(model_data.iloc[int(np.argmax(cooks))]["response_id"]),
        "n_above_4_over_n": int((cooks > 4 / len(model_data)).sum()),
        "resolution_block_incremental_r_squared": float(resolution_increment),
        "resolution_block_partial_r_squared": float(resolution_partial_r2),
    }
    write_json(tables / "ols_diagnostics.json", diagnostics)

    # Influence sensitivity after removing the largest Cook's-distance case.
    max_index = int(np.argmax(cooks))
    influence_sensitivity = model_data.drop(model_data.index[max_index]).copy()
    _, _, influence_fit, influence_robust = _fit_ols_hc3(
        influence_sensitivity, PRIMARY_PREDICTORS
    )
    pd.DataFrame(
        {
            "predictor": ["const"] + PRIMARY_PREDICTORS,
            "B": influence_robust.params,
            "p": influence_robust.pvalues,
        }
    ).to_csv(tables / "ols_remove_max_cook_sensitivity.csv", index=False)
    write_json(
        tables / "ols_remove_max_cook_summary.json",
        {
            "removed_response_id": diagnostics["max_cooks_response_id"],
            "n": int(influence_fit.nobs),
            "r_squared": float(influence_fit.rsquared),
            "adjusted_r_squared": float(influence_fit.rsquared_adj),
        },
    )

    # Huber robust-regression sensitivity on standardised predictors.
    scaler = StandardScaler()
    scaled_primary = scaler.fit_transform(model_data[PRIMARY_PREDICTORS].astype(float))
    huber = HuberRegressor(max_iter=10000).fit(scaled_primary, y)
    pd.DataFrame(
        {
            "predictor": PRIMARY_PREDICTORS,
            "standardised_huber_coefficient": huber.coef_,
        }
    ).to_csv(tables / "huber_sensitivity.csv", index=False)

    # Turnaround-applicable model.
    turnaround_data = model_data.dropna(subset=["turnaround_acceptability"]).copy()
    turnaround_predictors = PRIMARY_PREDICTORS + ["turnaround_acceptability"]
    _, _, turnaround_conventional, turnaround_robust = _fit_ols_hc3(
        turnaround_data, turnaround_predictors
    )
    turnaround_ci = turnaround_robust.conf_int()
    pd.DataFrame(
        {
            "predictor": ["const"] + turnaround_predictors,
            "B": turnaround_robust.params,
            "SE_HC3": turnaround_robust.bse,
            "CI_low": turnaround_ci[:, 0],
            "CI_high": turnaround_ci[:, 1],
            "p": turnaround_robust.pvalues,
        }
    ).to_csv(tables / "turnaround_subset_model.csv", index=False)
    write_json(
        tables / "turnaround_subset_summary.json",
        {
            "n": int(turnaround_conventional.nobs),
            "r_squared": float(turnaround_conventional.rsquared),
            "adjusted_r_squared": float(turnaround_conventional.rsquared_adj),
            "rmse": float(np.sqrt(np.mean(np.square(turnaround_conventional.resid)))),
        },
    )

    # Exploratory one-at-a-time extensions of the confirmatory model.
    extension_rows = []
    for predictor in EXPLORATORY_PREDICTORS:
        frame = model_data.dropna(subset=[predictor]).copy()
        base_fit = sm.OLS(
            frame["overall_satisfaction"].astype(float),
            sm.add_constant(frame[PRIMARY_PREDICTORS].astype(float), has_constant="add"),
        ).fit()
        extended_predictors = PRIMARY_PREDICTORS + [predictor]
        extended_x, _, extended_conventional, extended_robust = _fit_ols_hc3(
            frame, extended_predictors
        )
        index = list(extended_x.columns).index(predictor)
        extension_rows.append(
            {
                "predictor": predictor,
                "B": float(extended_robust.params[index]),
                "CI_low": float(extended_robust.conf_int()[index, 0]),
                "CI_high": float(extended_robust.conf_int()[index, 1]),
                "p": float(extended_robust.pvalues[index]),
                "incremental_r_squared": float(
                    extended_conventional.rsquared - base_fit.rsquared
                ),
                "extended_adjusted_r_squared": float(
                    extended_conventional.rsquared_adj
                ),
            }
        )
    pd.DataFrame(extension_rows).sort_values(
        "incremental_r_squared", ascending=False
    ).to_csv(tables / "exploratory_predictor_extensions.csv", index=False)

    # Wider elastic-net sensitivity model.
    wide_predictors = [
        "appointment_access",
        "waiting_acceptability",
        "communication_quality",
        "staff_helpfulness",
        "technical_knowledge",
        "explanation_clarity",
        "options_clarity",
        "cost_warranty_clarity",
        "process_ease",
        "resolution_partial",
        "resolution_unresolved",
    ]
    wide_data = model_data.dropna(subset=wide_predictors + ["overall_satisfaction"])
    wide_scaler = StandardScaler()
    wide_x = wide_scaler.fit_transform(wide_data[wide_predictors].astype(float))
    elastic_config = config["elastic_net"]
    elastic = ElasticNetCV(
        l1_ratio=elastic_config["l1_ratio"],
        alphas=np.logspace(
            elastic_config["alpha_min_log10"],
            elastic_config["alpha_max_log10"],
            int(elastic_config["n_alphas"]),
        ),
        cv=int(elastic_config["cv"]),
        random_state=int(config["random_seed"]),
        max_iter=int(elastic_config["max_iter"]),
    ).fit(wide_x, wide_data["overall_satisfaction"].astype(float))
    elastic_table = pd.DataFrame(
        {
            "predictor": wide_predictors,
            "elastic_net_coefficient": elastic.coef_,
            "absolute_coefficient": np.abs(elastic.coef_),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    elastic_table.to_csv(tables / "elastic_net_sensitivity.csv", index=False)
    write_json(
        tables / "elastic_net_configuration_selected.json",
        {
            "selected_alpha": float(elastic.alpha_),
            "selected_l1_ratio": float(elastic.l1_ratio_),
            "intercept": float(elastic.intercept_),
        },
    )

    # Figure 5: primary OLS effects with 95% HC3 intervals.
    plot = coefficient_table.loc[coefficient_table["predictor"].ne("const")].copy()
    plot = plot.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(
        plot["B"],
        np.arange(len(plot)),
        xerr=np.vstack([plot["B"] - plot["CI_low"], plot["CI_high"] - plot["B"]]),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(np.arange(len(plot)), plot["label"])
    ax.set_xlabel("Unstandardised coefficient B (95% HC3 CI)")
    ax.set_title("Primary explanatory model of overall satisfaction")
    fig.tight_layout()
    fig.savefig(figures / "figure_5_primary_ols_coefficients.png", dpi=300)
    plt.close(fig)

    # Supplementary regression diagnostics figure.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(conventional.fittedvalues, residuals, s=22)
    axes[0].axhline(0, linewidth=0.8)
    axes[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals versus fitted")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot")
    fig.tight_layout()
    fig.savefig(figures / "supplementary_regression_diagnostics.png", dpi=300)
    plt.close(fig)

    return {
        "primary": diagnostics,
        "turnaround": {
            "n": int(turnaround_conventional.nobs),
            "adjusted_r_squared": float(turnaround_conventional.rsquared_adj),
        },
    }
