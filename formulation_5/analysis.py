from __future__ import annotations

import numpy as np
import pandas as pd


ENERGY_DELTA_METRICS = {
    "delta_energy",
    "delta_energy_variance",
    "delta_expected_feasible_energy",
    "delta_feasible_energy_variance",
}

DEFAULT_SUMMARY_METRICS = [
    "delta_energy",
    "delta_energy_variance",
    "delta_p_ground",
    "delta_expected_feasible_energy",
    "delta_feasible_energy_variance",
    "delta_p_feas",
]


# return the metrics that are present in the dataframe.
def available_metrics(df: pd.DataFrame, metrics: list[str] | None = None) -> list[str]:
    candidates = DEFAULT_SUMMARY_METRICS if metrics is None else metrics
    return [metric for metric in candidates if metric in df.columns]


# identify whether a delta is favourable.
def favourable_mask(metric: str, values: np.ndarray) -> np.ndarray:
    if metric in ENERGY_DELTA_METRICS:
        return values < 0

    return values > 0


# return the mean of a column if it exists.
def _mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean()) if column in df.columns else float("nan")


# return the median of a column if it exists.
def _median(df: pd.DataFrame, column: str) -> float:
    return float(df[column].median()) if column in df.columns else float("nan")


# return the improvement rate of a delta column.
def _rate(df: pd.DataFrame, column: str, lower_is_better: bool) -> float:
    if column not in df.columns:
        return float("nan")

    values = df[column].dropna().to_numpy(dtype=float)

    if values.size == 0:
        return float("nan")

    if lower_is_better:
        return float(np.mean(values < 0))

    return float(np.mean(values > 0))


# return a row value if it exists.
def _value(row: pd.Series, column: str) -> float:
    if column in row.index and pd.notna(row[column]):
        return float(row[column])

    return float("nan")


# compute a bootstrap confidence interval.
def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20260531,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        sample = rng.choice(values, size=values.size, replace=True)
        stats[b] = statistic(sample)

    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


# compute a one-sided Wilcoxon p-value.
def wilcoxon_p_value(values: np.ndarray, metric: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0 or np.allclose(values, 0.0):
        return float("nan")

    try:
        from scipy.stats import wilcoxon

        alternative = "less" if metric in ENERGY_DELTA_METRICS else "greater"
        return float(wilcoxon(values, alternative=alternative, zero_method="wilcox").pvalue)

    except Exception:
        return float("nan")


# summarize delta metrics by chosen grouping columns.
def summarize_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = available_metrics(df)
    rows = []

    for group_key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        base = {col: value for col, value in zip(group_cols, group_key)}

        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)

            if values.size == 0:
                continue

            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "improvement_rate": float(np.mean(favourable_mask(metric, values))),
                    "wilcoxon_p_value": wilcoxon_p_value(values, metric),
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


# build confidence intervals by method.
def confidence_table(
    df: pd.DataFrame,
    metrics: list[str],
    n_boot: int = 10000,
) -> pd.DataFrame:
    rows = []
    group_cols = ["method_id", "degree", "restriction_type", "restriction_value_label"]

    for keys, group in df.groupby(group_cols, dropna=False):
        method_id, degree, restriction_type, restriction_value_label = keys

        for metric in available_metrics(group, metrics):
            values = group[metric].dropna().to_numpy(dtype=float)

            if values.size == 0:
                continue

            low, high = bootstrap_ci(values, n_boot=n_boot)

            rows.append(
                {
                    "method_id": method_id,
                    "degree": degree,
                    "restriction_type": restriction_type,
                    "restriction_value_label": restriction_value_label,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(values.size),
                }
            )

    return pd.DataFrame(rows)


# rank methods by aggregate performance.
def method_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method_id, group in df.groupby("method_id", dropna=False):
        delta_energy = group["delta_energy"].dropna().to_numpy(dtype=float)
        delta_p_ground = group["delta_p_ground"].dropna().to_numpy(dtype=float)

        row = {
            "method_id": method_id,
            "degree": int(group["degree"].iloc[0]),
            "restriction_type": group["restriction_type"].iloc[0],
            "restriction_value_label": group["restriction_value_label"].iloc[0],
            "num_instances": int(len(group)),
            "mean_delta_energy": float(np.mean(delta_energy)) if delta_energy.size else float("nan"),
            "median_delta_energy": float(np.median(delta_energy)) if delta_energy.size else float("nan"),
            "energy_improvement_rate": float(np.mean(delta_energy < 0)) if delta_energy.size else float("nan"),
            "mean_delta_p_ground": float(np.mean(delta_p_ground)) if delta_p_ground.size else float("nan"),
            "median_delta_p_ground": float(np.median(delta_p_ground)) if delta_p_ground.size else float("nan"),
            "pground_improvement_rate": float(np.mean(delta_p_ground > 0)) if delta_p_ground.size else float("nan"),
            "mean_feedback_time": _mean(group, "feedback_simulation_time"),
            "mean_beta_total_variation": _mean(group, "beta_total_variation"),
            "mean_beta_area": _mean(group, "beta_area"),
            "mean_stationary_selected_rate": _mean(group, "stationary_selected_rate"),
            "mean_endpoint_selected_rate": _mean(group, "endpoint_selected_rate"),
        }

        if "delta_energy_variance" in group.columns:
            row.update(
                {
                    "mean_delta_energy_variance": _mean(group, "delta_energy_variance"),
                    "median_delta_energy_variance": _median(group, "delta_energy_variance"),
                    "energy_variance_improvement_rate": _rate(
                        group,
                        "delta_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_expected_feasible_energy" in group.columns:
            row.update(
                {
                    "mean_delta_expected_feasible_energy": _mean(group, "delta_expected_feasible_energy"),
                    "median_delta_expected_feasible_energy": _median(group, "delta_expected_feasible_energy"),
                    "expected_feasible_energy_improvement_rate": _rate(
                        group,
                        "delta_expected_feasible_energy",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_feasible_energy_variance" in group.columns:
            row.update(
                {
                    "mean_delta_feasible_energy_variance": _mean(group, "delta_feasible_energy_variance"),
                    "median_delta_feasible_energy_variance": _median(group, "delta_feasible_energy_variance"),
                    "feasible_energy_variance_improvement_rate": _rate(
                        group,
                        "delta_feasible_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        rows.append(row)

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    ranking = ranking.sort_values(
        ["mean_delta_energy", "mean_delta_p_ground"],
        ascending=[True, False],
    )
    ranking["rank_by_energy"] = np.arange(1, len(ranking) + 1)

    ranking = ranking.sort_values(
        ["mean_delta_p_ground", "mean_delta_energy"],
        ascending=[False, True],
    )
    ranking["rank_by_p_ground"] = np.arange(1, len(ranking) + 1)

    if "mean_delta_expected_feasible_energy" in ranking.columns:
        ranking = ranking.sort_values(
            ["mean_delta_expected_feasible_energy", "mean_delta_energy"],
            ascending=[True, True],
        )
        ranking["rank_by_expected_feasible_energy"] = np.arange(1, len(ranking) + 1)

    return ranking.sort_values("rank_by_energy").reset_index(drop=True)


# select the best method for each instance.
def best_by_instance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["instance_id", "n", "p", "replicate"]

    for keys, group in df.groupby(group_cols, dropna=False):
        best_idx = group["feedback_final_expected_energy"].idxmin()
        best = group.loc[best_idx]

        row = {
            "instance_id": keys[0],
            "n": keys[1],
            "p": keys[2],
            "replicate": keys[3],
            "best_method_id": best["method_id"],
            "best_degree": int(best["degree"]),
            "best_restriction_type": best["restriction_type"],
            "best_restriction_value_label": best["restriction_value_label"],
            "base_final_expected_energy": float(best["base_final_expected_energy"]),
            "best_final_expected_energy": float(best["feedback_final_expected_energy"]),
            "best_delta_energy": float(best["delta_energy"]),
            "base_ground_state_probability": float(best["base_ground_state_probability"]),
            "best_ground_state_probability": float(best["feedback_ground_state_probability"]),
            "best_delta_p_ground": float(best["delta_p_ground"]),
        }

        if "base_energy_variance" in best.index:
            row.update(
                {
                    "base_energy_variance": _value(best, "base_energy_variance"),
                    "best_energy_variance": _value(best, "feedback_energy_variance"),
                    "best_delta_energy_variance": _value(best, "delta_energy_variance"),
                }
            )

        if "base_expected_feasible_energy" in best.index:
            row.update(
                {
                    "base_expected_feasible_energy": _value(best, "base_expected_feasible_energy"),
                    "best_expected_feasible_energy": _value(best, "feedback_expected_feasible_energy"),
                    "best_delta_expected_feasible_energy": _value(best, "delta_expected_feasible_energy"),
                }
            )

        if "base_feasible_energy_variance" in best.index:
            row.update(
                {
                    "base_feasible_energy_variance": _value(best, "base_feasible_energy_variance"),
                    "best_feasible_energy_variance": _value(best, "feedback_feasible_energy_variance"),
                    "best_delta_feasible_energy_variance": _value(best, "delta_feasible_energy_variance"),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# summarize results by polynomial degree.
def degree_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for degree, group in df.groupby("degree", dropna=False):
        row = {
            "degree": int(degree),
            "num_results": int(len(group)),
            "mean_delta_energy": _mean(group, "delta_energy"),
            "median_delta_energy": _median(group, "delta_energy"),
            "energy_improvement_rate": _rate(group, "delta_energy", lower_is_better=True),
            "mean_delta_p_ground": _mean(group, "delta_p_ground"),
            "median_delta_p_ground": _median(group, "delta_p_ground"),
            "pground_improvement_rate": _rate(group, "delta_p_ground", lower_is_better=False),
            "mean_feedback_time": _mean(group, "feedback_simulation_time"),
        }

        if "delta_energy_variance" in group.columns:
            row.update(
                {
                    "mean_delta_energy_variance": _mean(group, "delta_energy_variance"),
                    "median_delta_energy_variance": _median(group, "delta_energy_variance"),
                    "energy_variance_improvement_rate": _rate(
                        group,
                        "delta_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_expected_feasible_energy" in group.columns:
            row.update(
                {
                    "mean_delta_expected_feasible_energy": _mean(group, "delta_expected_feasible_energy"),
                    "median_delta_expected_feasible_energy": _median(group, "delta_expected_feasible_energy"),
                    "expected_feasible_energy_improvement_rate": _rate(
                        group,
                        "delta_expected_feasible_energy",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_feasible_energy_variance" in group.columns:
            row.update(
                {
                    "mean_delta_feasible_energy_variance": _mean(group, "delta_feasible_energy_variance"),
                    "median_delta_feasible_energy_variance": _median(group, "delta_feasible_energy_variance"),
                    "feasible_energy_variance_improvement_rate": _rate(
                        group,
                        "delta_feasible_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)


# select the best aggregate method within each degree.
def best_method_by_degree_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for degree, degree_df in df.groupby("degree", dropna=False):
        agg_spec = {
            "mean_delta_energy": ("delta_energy", "mean"),
            "median_delta_energy": ("delta_energy", "median"),
            "mean_delta_p_ground": ("delta_p_ground", "mean"),
            "median_delta_p_ground": ("delta_p_ground", "median"),
            "mean_feedback_time": ("feedback_simulation_time", "mean"),
            "num_instances": ("instance_id", "count"),
        }

        if "delta_energy_variance" in degree_df.columns:
            agg_spec.update(
                {
                    "mean_delta_energy_variance": ("delta_energy_variance", "mean"),
                    "median_delta_energy_variance": ("delta_energy_variance", "median"),
                }
            )

        if "delta_expected_feasible_energy" in degree_df.columns:
            agg_spec.update(
                {
                    "mean_delta_expected_feasible_energy": ("delta_expected_feasible_energy", "mean"),
                    "median_delta_expected_feasible_energy": ("delta_expected_feasible_energy", "median"),
                }
            )

        if "delta_feasible_energy_variance" in degree_df.columns:
            agg_spec.update(
                {
                    "mean_delta_feasible_energy_variance": ("delta_feasible_energy_variance", "mean"),
                    "median_delta_feasible_energy_variance": ("delta_feasible_energy_variance", "median"),
                }
            )

        grouped = (
            degree_df.groupby(
                ["method_id", "restriction_type", "restriction_value_label"],
                dropna=False,
            )
            .agg(**agg_spec)
            .reset_index()
        )

        if grouped.empty:
            continue

        best = grouped.sort_values(
            ["mean_delta_energy", "mean_delta_p_ground"],
            ascending=[True, False],
        ).iloc[0]

        row = best.to_dict()
        row["degree"] = int(degree)

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# select the best method for each instance and degree.
def best_by_instance_by_degree_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["instance_id", "n", "p", "replicate", "degree"]

    for keys, group in df.groupby(group_cols, dropna=False):
        best_idx = group["feedback_final_expected_energy"].idxmin()
        best = group.loc[best_idx]

        row = {
            "instance_id": keys[0],
            "n": keys[1],
            "p": keys[2],
            "replicate": keys[3],
            "degree": int(keys[4]),
            "best_method_id": best["method_id"],
            "best_restriction_type": best["restriction_type"],
            "best_restriction_value_label": best["restriction_value_label"],
            "best_restriction_label": best.get("restriction_label", best["method_id"]),
            "base_final_expected_energy": float(best["base_final_expected_energy"]),
            "best_final_expected_energy": float(best["feedback_final_expected_energy"]),
            "best_delta_energy": float(best["delta_energy"]),
            "base_ground_state_probability": float(best["base_ground_state_probability"]),
            "best_ground_state_probability": float(best["feedback_ground_state_probability"]),
            "best_delta_p_ground": float(best["delta_p_ground"]),
        }

        if "base_energy_variance" in best.index:
            row.update(
                {
                    "base_energy_variance": _value(best, "base_energy_variance"),
                    "best_energy_variance": _value(best, "feedback_energy_variance"),
                    "best_delta_energy_variance": _value(best, "delta_energy_variance"),
                }
            )

        if "base_expected_feasible_energy" in best.index:
            row.update(
                {
                    "base_expected_feasible_energy": _value(best, "base_expected_feasible_energy"),
                    "best_expected_feasible_energy": _value(best, "feedback_expected_feasible_energy"),
                    "best_delta_expected_feasible_energy": _value(best, "delta_expected_feasible_energy"),
                }
            )

        if "base_feasible_energy_variance" in best.index:
            row.update(
                {
                    "base_feasible_energy_variance": _value(best, "base_feasible_energy_variance"),
                    "best_feasible_energy_variance": _value(best, "feedback_feasible_energy_variance"),
                    "best_delta_feasible_energy_variance": _value(best, "delta_feasible_energy_variance"),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# summarize the best method selected for each degree.
def q_section_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for degree, group in df.groupby("degree", dropna=False):
        best_idx = group.groupby("method_id")["delta_energy"].mean().idxmin()
        best_group = group[group["method_id"] == best_idx]

        row = {
            "degree": int(degree),
            "best_method_id": best_idx,
            "best_restriction_label": str(best_group["restriction_label"].iloc[0])
            if "restriction_label" in best_group
            else best_idx,
            "mean_delta_energy": _mean(best_group, "delta_energy"),
            "median_delta_energy": _median(best_group, "delta_energy"),
            "energy_improvement_rate": _rate(best_group, "delta_energy", lower_is_better=True),
            "mean_delta_p_ground": _mean(best_group, "delta_p_ground"),
            "median_delta_p_ground": _median(best_group, "delta_p_ground"),
            "pground_improvement_rate": _rate(best_group, "delta_p_ground", lower_is_better=False),
            "mean_feedback_time": _mean(best_group, "feedback_simulation_time"),
        }

        if "delta_energy_variance" in best_group.columns:
            row.update(
                {
                    "mean_delta_energy_variance": _mean(best_group, "delta_energy_variance"),
                    "median_delta_energy_variance": _median(best_group, "delta_energy_variance"),
                    "energy_variance_improvement_rate": _rate(
                        best_group,
                        "delta_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_expected_feasible_energy" in best_group.columns:
            row.update(
                {
                    "mean_delta_expected_feasible_energy": _mean(best_group, "delta_expected_feasible_energy"),
                    "median_delta_expected_feasible_energy": _median(best_group, "delta_expected_feasible_energy"),
                    "expected_feasible_energy_improvement_rate": _rate(
                        best_group,
                        "delta_expected_feasible_energy",
                        lower_is_better=True,
                    ),
                }
            )

        if "delta_feasible_energy_variance" in best_group.columns:
            row.update(
                {
                    "mean_delta_feasible_energy_variance": _mean(best_group, "delta_feasible_energy_variance"),
                    "median_delta_feasible_energy_variance": _median(best_group, "delta_feasible_energy_variance"),
                    "feasible_energy_variance_improvement_rate": _rate(
                        best_group,
                        "delta_feasible_energy_variance",
                        lower_is_better=True,
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)
