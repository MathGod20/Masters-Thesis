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


# compute a bootstrap confidence interval.
def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20260609,
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


# build confidence intervals for paired deltas.
def confidence_table(
    df: pd.DataFrame,
    metrics: list[str],
    n_boot: int = 10000,
) -> pd.DataFrame:
    rows = []

    group_cols = ["mu"] if "mu" in df.columns else ["method"]

    for group_key, group in df.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            group_key = group_key[0]

        for metric in available_metrics(group, metrics):
            values = group[metric].dropna().to_numpy(dtype=float)

            if values.size == 0:
                continue

            low, high = bootstrap_ci(values, n_boot=n_boot)

            rows.append(
                {
                    "method": f"block_mu_{group_key}" if "mu" in df.columns else group_key,
                    "mu": int(group_key) if "mu" in df.columns and pd.notna(group_key) else np.nan,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "n": int(values.size),
                    "improvement_rate": float(np.mean(favourable_mask(metric, values))),
                    "wilcoxon_p_value": wilcoxon_p_value(values, metric),
                }
            )

    return pd.DataFrame(rows)


# summarize block-mu performance by mu.
def overall_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mu, group in df.groupby("mu", dropna=False):
        rows.append(
            {
                "method": f"block_mu_{int(mu)}",
                "mu": int(mu),
                "num_instances": int(len(group)),
                "mean_delta_energy": _mean(group, "delta_energy"),
                "median_delta_energy": _median(group, "delta_energy"),
                "energy_improvement_rate": _rate(group, "delta_energy", lower_is_better=True),
                "mean_delta_energy_variance": _mean(group, "delta_energy_variance"),
                "median_delta_energy_variance": _median(group, "delta_energy_variance"),
                "energy_variance_improvement_rate": _rate(group, "delta_energy_variance", lower_is_better=True),
                "mean_delta_p_ground": _mean(group, "delta_p_ground"),
                "median_delta_p_ground": _median(group, "delta_p_ground"),
                "pground_improvement_rate": _rate(group, "delta_p_ground", lower_is_better=False),
                "mean_delta_expected_feasible_energy": _mean(group, "delta_expected_feasible_energy"),
                "median_delta_expected_feasible_energy": _median(group, "delta_expected_feasible_energy"),
                "expected_feasible_energy_improvement_rate": _rate(
                    group,
                    "delta_expected_feasible_energy",
                    lower_is_better=True,
                ),
                "mean_delta_feasible_energy_variance": _mean(group, "delta_feasible_energy_variance"),
                "median_delta_feasible_energy_variance": _median(group, "delta_feasible_energy_variance"),
                "feasible_energy_variance_improvement_rate": _rate(
                    group,
                    "delta_feasible_energy_variance",
                    lower_is_better=True,
                ),
                "mean_delta_p_feas": _mean(group, "delta_p_feas"),
                "median_delta_p_feas": _median(group, "delta_p_feas"),
                "pfeas_improvement_rate": _rate(group, "delta_p_feas", lower_is_better=False),
                "mean_base_simulation_time": _mean(group, "base_simulation_time"),
                "mean_block_simulation_time": _mean(group, "block_simulation_time"),
                "mean_num_blocks": _mean(group, "num_blocks"),
                "mean_block_size": _mean(group, "mean_block_size"),
                "mean_max_block_size": _mean(group, "max_block_size"),
                "mean_block_valid_probability": _mean(group, "block_final_block_valid_probability"),
            }
        )

    return pd.DataFrame(rows)
