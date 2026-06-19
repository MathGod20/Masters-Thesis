from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import (
    best_by_instance_by_degree_table,
    best_by_instance_table,
    best_method_by_degree_table,
    degree_summary_table,
    method_ranking_table,
    q_section_summary_table,
)

RESTRICTION_ORDER = {"absolute": 0, "relative": 1, "none": 2}


def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _ordered_methods(results_df: pd.DataFrame, max_methods: int | None = None) -> list[str]:
    if "delta_energy" in results_df.columns:
        order = results_df.groupby("method_id")["delta_energy"].mean().sort_values().index.tolist()
    else:
        order = sorted(results_df["method_id"].dropna().unique().tolist())
    if max_methods is not None:
        return order[:max_methods]
    return order


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def _finite_pair_df(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if not _has_columns(df, [x_col, y_col]):
        return pd.DataFrame()
    out = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _restriction_value(row: pd.Series) -> float:
    if "restriction_value" in row.index and pd.notna(row["restriction_value"]):
        return float(row["restriction_value"])
    text = str(row.get("restriction_value_label", ""))
    if text == "none":
        return float("inf")
    try:
        return float(text.replace("p", "."))
    except Exception:
        return float("inf")


def _method_meta(results_df: pd.DataFrame, degree: int | None = None) -> pd.DataFrame:
    cols = ["method_id", "degree", "restriction_type", "restriction_value_label", "restriction_label"]
    cols = [col for col in cols if col in results_df.columns]
    meta = results_df[cols].drop_duplicates().copy()
    if degree is not None:
        meta = meta[meta["degree"] == degree]
    if meta.empty:
        return meta
    meta["restriction_sort"] = meta["restriction_type"].map(RESTRICTION_ORDER).fillna(99)
    meta["value_sort"] = meta.apply(_restriction_value, axis=1)
    meta = meta.sort_values(["degree", "restriction_sort", "value_sort", "method_id"])
    return meta


def _method_label_map(results_df: pd.DataFrame) -> dict[str, str]:
    meta = _method_meta(results_df)
    mapping = {}
    for _, row in meta.iterrows():
        label = row.get("restriction_label", row["method_id"])
        mapping[row["method_id"]] = str(label)
    return mapping


def _degree_methods(results_df: pd.DataFrame, degree: int) -> list[str]:
    return _method_meta(results_df, degree=degree)["method_id"].tolist()


def _best_method_for_degree(results_df: pd.DataFrame, degree: int) -> str | None:
    degree_df = results_df[results_df["degree"] == degree]
    if degree_df.empty:
        return None
    return degree_df.groupby("method_id")["delta_energy"].mean().sort_values().index[0]


def plot_mean_energy_trajectory_top_methods(history_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: str | Path, top_n: int = 5) -> None:
    top_methods = _ordered_methods(results_df, top_n)
    keep = ["baseline"] + top_methods
    mean_df = history_df[history_df["method"].isin(keep)].groupby(["method", "step"], as_index=False)["expected_energy"].mean()
    plt.figure(figsize=(9, 5))
    for method, group in mean_df.groupby("method"):
        linewidth = 2.4 if method == "baseline" else 1.4
        plt.plot(group["step"], group["expected_energy"], label=method, linewidth=linewidth)
    plt.xlabel("Annealing step")
    plt.ylabel("Mean expected energy")
    plt.title("Mean energy trajectory: baseline and best feedback variants")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / "overview_energy_trajectory_top_methods.png")


def plot_delta_energy_by_method(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    methods = _ordered_methods(results_df)
    data = [results_df.loc[results_df["method_id"] == method, "delta_energy"].dropna().to_numpy() for method in methods]
    plt.figure(figsize=(10, max(5, 0.32 * len(methods))))
    plt.boxplot(data, labels=methods, vert=False)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"$\Delta E = E_{feedback} - E_{base}$")
    plt.title("Paired final-energy difference by feedback variant")
    save_figure(Path(out_dir) / "overview_delta_energy_by_method.png")


def plot_delta_pground_by_method(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    methods = _ordered_methods(results_df)
    data = [results_df.loc[results_df["method_id"] == method, "delta_p_ground"].dropna().to_numpy() for method in methods]
    plt.figure(figsize=(10, max(5, 0.32 * len(methods))))
    plt.boxplot(data, labels=methods, vert=False)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"$\Delta P_{ground}$")
    plt.title("Paired ground-state probability difference by feedback variant")
    save_figure(Path(out_dir) / "overview_delta_pground_by_method.png")


def plot_method_heatmap(results_df: pd.DataFrame, out_dir: str | Path, metric: str = "delta_energy") -> None:
    if metric not in results_df.columns:
        return
    title_map = {
        "delta_energy": "Mean final-energy difference",
        "delta_energy_variance": "Mean energy-variance difference",
        "delta_expected_feasible_energy": "Mean expected feasible-energy difference",
        "delta_feasible_energy_variance": "Mean feasible energy-variance difference",
        "delta_p_ground": "Mean ground-state probability difference",
        "delta_p_feas": "Mean feasibility-probability difference",
    }
    suffix_map = {
        "delta_energy": "energy",
        "delta_energy_variance": "energy_variance",
        "delta_expected_feasible_energy": "expected_feasible_energy",
        "delta_feasible_energy_variance": "feasible_energy_variance",
        "delta_p_ground": "pground",
        "delta_p_feas": "pfeas",
    }
    title_metric = title_map.get(metric, f"Mean {metric} difference")
    table = results_df.pivot_table(index="restriction_label", columns="degree", values=metric, aggfunc="mean")
    if table.empty:
        return
    plt.figure(figsize=(7, max(4, 0.36 * len(table))))
    image = plt.imshow(table.values, aspect="auto")
    plt.colorbar(image, label=title_metric)
    plt.xticks(np.arange(len(table.columns)), [str(c) for c in table.columns])
    plt.yticks(np.arange(len(table.index)), table.index)
    plt.xlabel("Polynomial degree q")
    plt.ylabel("Restriction")
    plt.title(f"{title_metric} by degree and restriction")
    suffix = suffix_map.get(metric, metric)
    save_figure(Path(out_dir) / f"overview_{suffix}_heatmap.png")


def plot_delta_metric_by_degree(results_df: pd.DataFrame, out_dir: str | Path, metric: str, ylabel: str, filename: str) -> None:
    if metric not in results_df.columns:
        return
    degrees = sorted(results_df["degree"].dropna().unique())
    data, labels = [], []
    for degree in degrees:
        values = results_df.loc[results_df["degree"] == degree, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(f"q={int(degree)}")
    if not data:
        return
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Polynomial degree")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by polynomial degree")
    save_figure(Path(out_dir) / filename)


def plot_pground_scatter_best(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    ranking = results_df.groupby("method_id")["delta_energy"].mean().sort_values()
    if ranking.empty:
        return
    best_method = ranking.index[0]
    df = results_df[results_df["method_id"] == best_method]
    plt.figure(figsize=(6, 6))
    plt.scatter(df["base_ground_state_probability"], df["feedback_ground_state_probability"], alpha=0.7)
    max_value = max(df["base_ground_state_probability"].max(), df["feedback_ground_state_probability"].max(), 1e-12)
    plt.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1)
    plt.xlabel("Baseline ground-state probability")
    plt.ylabel("Feedback ground-state probability")
    plt.title(f"Ground-state probability comparison: {best_method}")
    save_figure(Path(out_dir) / "overview_pground_scatter_best_method.png")


def plot_runtime_vs_improvement(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    summary = results_df.groupby(["method_id", "degree"], as_index=False).agg(
        mean_delta_energy=("delta_energy", "mean"),
        mean_feedback_time=("feedback_simulation_time", "mean"),
    )
    if summary.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(summary["mean_feedback_time"], summary["mean_delta_energy"])
    for _, row in summary.iterrows():
        plt.annotate(row["method_id"], (row["mean_feedback_time"], row["mean_delta_energy"]), fontsize=7, alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Mean feedback simulation time in seconds")
    plt.ylabel("Mean delta energy")
    plt.title("Improvement versus computational overhead")
    save_figure(Path(out_dir) / "overview_runtime_vs_improvement.png")


def plot_mean_beta_schedule_top_methods(schedule_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: str | Path, top_n: int = 5) -> None:
    top_methods = _ordered_methods(results_df, top_n)
    keep = ["baseline"] + top_methods
    mean_df = schedule_df[schedule_df["method"].isin(keep)].groupby(["method", "step"], as_index=False)["beta"].mean()
    plt.figure(figsize=(9, 5))
    for method, group in mean_df.groupby("method"):
        linewidth = 2.4 if method == "baseline" else 1.4
        plt.plot(group["step"], group["beta"], label=method, linewidth=linewidth)
    plt.xlabel("Annealing step")
    plt.ylabel(r"Mean mixer coefficient $\beta_k$")
    plt.title("Mean mixer schedules: best feedback variants")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / "overview_beta_schedule_top_methods.png")


def plot_winner_counts(best_df: pd.DataFrame, out_dir: str | Path) -> None:
    counts = best_df["best_method_id"].value_counts().sort_values(ascending=True)
    if counts.empty:
        return
    plt.figure(figsize=(9, max(4, 0.35 * len(counts))))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Number of instances")
    plt.title("Best feedback variant by instance")
    save_figure(Path(out_dir) / "overview_winner_counts.png")


def plot_best_delta_energy_by_n(best_df: pd.DataFrame, out_dir: str | Path) -> None:
    groups = list(best_df.groupby("n"))
    if not groups:
        return
    data = [group["best_delta_energy"].dropna().to_numpy() for _, group in groups]
    labels = [str(key) for key, _ in groups]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph size n")
    plt.ylabel("Best feedback delta energy")
    plt.title("Best feedback improvement by graph size")
    save_figure(Path(out_dir) / "overview_best_delta_energy_by_n.png")


def plot_best_delta_pground_by_p(best_df: pd.DataFrame, out_dir: str | Path) -> None:
    groups = list(best_df.groupby("p"))
    if not groups:
        return
    data = [group["best_delta_p_ground"].dropna().to_numpy() for _, group in groups]
    labels = [str(key) for key, _ in groups]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph density p")
    plt.ylabel("Best feedback delta ground-state probability")
    plt.title("Best feedback probability improvement by density")
    save_figure(Path(out_dir) / "overview_best_delta_pground_by_p.png")


def plot_mean_energy_trajectory_by_degree(history_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    methods = _degree_methods(results_df, degree)
    if not methods:
        return
    labels = _method_label_map(results_df)
    keep = ["baseline"] + methods
    mean_df = history_df[history_df["method"].isin(keep)].groupby(["method", "step"], as_index=False)["expected_energy"].mean()
    plt.figure(figsize=(9, 5))
    for method, group in mean_df.groupby("method"):
        linewidth = 2.6 if method == "baseline" else 1.4
        label = "baseline" if method == "baseline" else labels.get(method, method)
        plt.plot(group["step"], group["expected_energy"], label=label, linewidth=linewidth)
    plt.xlabel("Annealing step")
    plt.ylabel("Mean expected energy")
    plt.title(f"Mean energy trajectory for q={degree}")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / f"q{degree}_mean_energy_trajectory.png")


def plot_mean_beta_schedule_by_degree(schedule_df: pd.DataFrame, results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    methods = _degree_methods(results_df, degree)
    if not methods:
        return
    labels = _method_label_map(results_df)
    keep = ["baseline"] + methods
    mean_df = schedule_df[schedule_df["method"].isin(keep)].groupby(["method", "step"], as_index=False)["beta"].mean()
    plt.figure(figsize=(9, 5))
    for method, group in mean_df.groupby("method"):
        linewidth = 2.6 if method == "baseline" else 1.4
        label = "baseline" if method == "baseline" else labels.get(method, method)
        plt.plot(group["step"], group["beta"], label=label, linewidth=linewidth)
    plt.xlabel("Annealing step")
    plt.ylabel(r"Mean mixer coefficient $\beta_k$")
    plt.title(f"Mean mixer schedule for q={degree}")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / f"q{degree}_mean_beta_schedule.png")


def plot_delta_metric_by_restriction_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int, metric: str, xlabel: str, filename: str) -> None:
    if metric not in results_df.columns:
        return
    methods = _degree_methods(results_df, degree)
    if not methods:
        return
    labels = _method_label_map(results_df)
    data, plot_labels = [], []
    for method in methods:
        values = results_df.loc[results_df["method_id"] == method, metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            plot_labels.append(labels.get(method, method))
    if not data:
        return
    plt.figure(figsize=(9, max(4.5, 0.5 * len(data))))
    plt.boxplot(data, labels=plot_labels, vert=False)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.title(f"{xlabel} by restriction for q={degree}")
    save_figure(Path(out_dir) / filename)


def plot_pground_scatter_best_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    best_method = _best_method_for_degree(results_df, degree)
    if best_method is None:
        return
    df = results_df[results_df["method_id"] == best_method]
    label = _method_label_map(results_df).get(best_method, best_method)
    plt.figure(figsize=(6, 6))
    plt.scatter(df["base_ground_state_probability"], df["feedback_ground_state_probability"], alpha=0.7)
    max_value = max(df["base_ground_state_probability"].max(), df["feedback_ground_state_probability"].max(), 1e-12)
    plt.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1)
    plt.xlabel("Baseline ground-state probability")
    plt.ylabel("Feedback ground-state probability")
    plt.title(f"Ground-state probability: q={degree}, {label}")
    save_figure(Path(out_dir) / f"q{degree}_best_method_pground_scatter.png")


def plot_best_method_delta_energy_by_density_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    best_method = _best_method_for_degree(results_df, degree)
    if best_method is None:
        return
    df = results_df[results_df["method_id"] == best_method]
    groups = list(df.groupby("p"))
    if not groups:
        return
    data = [group["delta_energy"].dropna().to_numpy() for _, group in groups]
    labels = [str(key) for key, _ in groups]
    label = _method_label_map(results_df).get(best_method, best_method)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph density p")
    plt.ylabel(r"$\Delta E$")
    plt.title(f"Final-energy difference by density: q={degree}, {label}")
    save_figure(Path(out_dir) / f"q{degree}_best_method_delta_energy_by_density.png")


def plot_best_method_delta_pground_by_density_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    best_method = _best_method_for_degree(results_df, degree)
    if best_method is None:
        return
    df = results_df[results_df["method_id"] == best_method]
    groups = list(df.groupby("p"))
    if not groups:
        return
    data = [group["delta_p_ground"].dropna().to_numpy() for _, group in groups]
    labels = [str(key) for key, _ in groups]
    label = _method_label_map(results_df).get(best_method, best_method)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph density p")
    plt.ylabel(r"$\Delta P_{ground}$")
    plt.title(f"Ground-state probability difference by density: q={degree}, {label}")
    save_figure(Path(out_dir) / f"q{degree}_best_method_delta_pground_by_density.png")


def plot_best_restriction_counts_for_degree(best_by_degree_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    df = best_by_degree_df[best_by_degree_df["degree"] == degree]
    if df.empty:
        return
    counts = df["best_restriction_label"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(8, max(4, 0.45 * len(counts))))
    plt.barh(counts.index, counts.values)
    plt.xlabel("Number of instances")
    plt.title(f"Best restriction by instance for q={degree}")
    save_figure(Path(out_dir) / f"q{degree}_best_restriction_counts.png")


def plot_delta_feasible_energy_by_method(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "delta_expected_feasible_energy" not in results_df.columns:
        return
    methods = _ordered_methods(results_df)
    data, labels = [], []
    for method in methods:
        values = results_df.loc[results_df["method_id"] == method, "delta_expected_feasible_energy"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(method)
    if not data:
        return
    plt.figure(figsize=(10, max(5, 0.32 * len(labels))))
    plt.boxplot(data, labels=labels, vert=False)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"$\Delta E_{feas}$")
    plt.title("Paired expected feasible-energy difference by feedback variant")
    save_figure(Path(out_dir) / "overview_delta_expected_feasible_energy_by_method.png")


def _identity_axis_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        padding = max(1.0, abs(low) * 0.05)
    else:
        padding = 0.05 * (high - low)
    return low - padding, high + padding


def plot_feasible_energy_scatter_best(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "delta_expected_feasible_energy" not in results_df.columns:
        return
    ranking = results_df.groupby("method_id")["delta_energy"].mean().sort_values()
    if ranking.empty:
        return
    best_method = ranking.index[0]
    df = results_df[results_df["method_id"] == best_method]
    pairs = _finite_pair_df(df, "base_expected_feasible_energy", "feedback_expected_feasible_energy")
    if pairs.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(pairs["base_expected_feasible_energy"], pairs["feedback_expected_feasible_energy"], alpha=0.7)
    low, high = _identity_axis_limits(pairs.to_numpy().ravel())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
    plt.xlabel("Baseline expected feasible energy")
    plt.ylabel("Feedback expected feasible energy")
    plt.title(f"Expected feasible energy comparison: {best_method}")
    save_figure(Path(out_dir) / "overview_expected_feasible_energy_scatter_best_method.png")


def plot_feasible_energy_scatter_best_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    if "delta_expected_feasible_energy" not in results_df.columns:
        return
    best_method = _best_method_for_degree(results_df, degree)
    if best_method is None:
        return
    df = results_df[results_df["method_id"] == best_method]
    pairs = _finite_pair_df(df, "base_expected_feasible_energy", "feedback_expected_feasible_energy")
    if pairs.empty:
        return
    label = _method_label_map(results_df).get(best_method, best_method)
    plt.figure(figsize=(6, 6))
    plt.scatter(pairs["base_expected_feasible_energy"], pairs["feedback_expected_feasible_energy"], alpha=0.7)
    low, high = _identity_axis_limits(pairs.to_numpy().ravel())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
    plt.xlabel("Baseline expected feasible energy")
    plt.ylabel("Feedback expected feasible energy")
    plt.title(f"Expected feasible energy: q={degree}, {label}")
    save_figure(Path(out_dir) / f"q{degree}_best_method_expected_feasible_energy_scatter.png")


def plot_best_delta_feasible_energy_by_n(best_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "best_delta_expected_feasible_energy" not in best_df.columns:
        return
    groups = list(best_df.groupby("n"))
    if not groups:
        return
    data, labels = [], []
    for key, group in groups:
        values = group["best_delta_expected_feasible_energy"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(str(key))
    if not data:
        return
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph size n")
    plt.ylabel("Best feedback delta expected feasible energy")
    plt.title("Best feedback expected feasible-energy improvement by graph size")
    save_figure(Path(out_dir) / "overview_best_delta_expected_feasible_energy_by_n.png")


def plot_best_method_delta_feasible_energy_by_density_for_degree(results_df: pd.DataFrame, out_dir: str | Path, degree: int) -> None:
    if "delta_expected_feasible_energy" not in results_df.columns:
        return
    best_method = _best_method_for_degree(results_df, degree)
    if best_method is None:
        return
    df = results_df[results_df["method_id"] == best_method]
    groups = list(df.groupby("p"))
    if not groups:
        return
    data, labels = [], []
    for key, group in groups:
        values = group["delta_expected_feasible_energy"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(str(key))
    if not data:
        return
    label = _method_label_map(results_df).get(best_method, best_method)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Graph density p")
    plt.ylabel(r"$\Delta E_{feas}$")
    plt.title(f"Expected feasible-energy difference by density: q={degree}, {label}")
    save_figure(Path(out_dir) / f"q{degree}_best_method_delta_expected_feasible_energy_by_density.png")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_formulation5_experiment.py first, or copy your saved results into this package."
        )


def main() -> None:
    results_dir = Path("results")
    tables_dir = results_dir / "tables"
    raw_dir = results_dir / "raw"
    plots_dir = results_dir / "plots"
    overview_dir = plots_dir / "overview"
    by_degree_dir = plots_dir / "by_degree"

    plots_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    by_degree_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "formulation5_per_instance_results.csv"
    histories_path = raw_dir / "formulation5_energy_histories.csv"
    schedules_path = raw_dir / "formulation5_beta_schedules.csv"
    for path in [results_path, histories_path, schedules_path]:
        require_file(path)

    results = pd.read_csv(results_path)
    histories = pd.read_csv(histories_path)
    schedules = pd.read_csv(schedules_path)

    # rebuild interpretation tables from saved results.
    ranking = method_ranking_table(results)
    ranking.to_csv(tables_dir / "formulation5_method_ranking.csv", index=False)

    q_summary = q_section_summary_table(results)
    q_summary.to_csv(tables_dir / "formulation5_q_section_summary.csv", index=False)

    degree_summary = degree_summary_table(results)
    degree_summary.to_csv(tables_dir / "formulation5_degree_summary.csv", index=False)

    best_method_by_degree = best_method_by_degree_table(results)
    best_method_by_degree.to_csv(tables_dir / "formulation5_best_method_by_degree.csv", index=False)

    best = best_by_instance_table(results)
    best.to_csv(tables_dir / "formulation5_best_by_instance.csv", index=False)

    best_by_degree = best_by_instance_by_degree_table(results)
    best_by_degree.to_csv(tables_dir / "formulation5_best_by_instance_by_degree.csv", index=False)

    # create overview plots.
    plot_mean_energy_trajectory_top_methods(histories, results, overview_dir, top_n=5)
    plot_delta_energy_by_method(results, overview_dir)
    plot_delta_feasible_energy_by_method(results, overview_dir)
    plot_delta_pground_by_method(results, overview_dir)
    plot_method_heatmap(results, overview_dir, metric="delta_energy")
    plot_method_heatmap(results, overview_dir, metric="delta_energy_variance")
    plot_method_heatmap(results, overview_dir, metric="delta_expected_feasible_energy")
    plot_method_heatmap(results, overview_dir, metric="delta_feasible_energy_variance")
    plot_method_heatmap(results, overview_dir, metric="delta_p_ground")
    plot_delta_metric_by_degree(
        results,
        overview_dir,
        metric="delta_energy",
        ylabel=r"$\Delta E$",
        filename="overview_delta_energy_by_degree.png",
    )
    plot_delta_metric_by_degree(
        results,
        overview_dir,
        metric="delta_energy_variance",
        ylabel=r"$\Delta \mathrm{Var}(E)$",
        filename="overview_delta_energy_variance_by_degree.png",
    )
    plot_delta_metric_by_degree(
        results,
        overview_dir,
        metric="delta_p_ground",
        ylabel=r"$\Delta P_{ground}$",
        filename="overview_delta_pground_by_degree.png",
    )
    plot_delta_metric_by_degree(
        results,
        overview_dir,
        metric="delta_expected_feasible_energy",
        ylabel=r"$\Delta E_{feas}$",
        filename="overview_delta_expected_feasible_energy_by_degree.png",
    )
    plot_delta_metric_by_degree(
        results,
        overview_dir,
        metric="delta_feasible_energy_variance",
        ylabel=r"$\Delta \mathrm{Var}(E_{feas})$",
        filename="overview_delta_feasible_energy_variance_by_degree.png",
    )
    plot_pground_scatter_best(results, overview_dir)
    plot_feasible_energy_scatter_best(results, overview_dir)
    plot_runtime_vs_improvement(results, overview_dir)
    plot_mean_beta_schedule_top_methods(schedules, results, overview_dir, top_n=5)
    plot_winner_counts(best, overview_dir)
    plot_best_delta_energy_by_n(best, overview_dir)
    plot_best_delta_feasible_energy_by_n(best, overview_dir)
    plot_best_delta_pground_by_p(best, overview_dir)

    # create degree-specific plots.
    for degree in sorted(results["degree"].dropna().unique()):
        degree = int(degree)
        q_dir = by_degree_dir / f"q{degree}"
        q_dir.mkdir(parents=True, exist_ok=True)
        plot_mean_energy_trajectory_by_degree(histories, results, q_dir, degree)
        plot_mean_beta_schedule_by_degree(schedules, results, q_dir, degree)
        plot_delta_metric_by_restriction_for_degree(
            results,
            q_dir,
            degree,
            metric="delta_energy",
            xlabel=r"$\Delta E = E_{feedback} - E_{base}$",
            filename=f"q{degree}_delta_energy_by_restriction.png",
        )
        plot_delta_metric_by_restriction_for_degree(
            results,
            q_dir,
            degree,
            metric="delta_energy_variance",
            xlabel=r"$\Delta \mathrm{Var}(E)$",
            filename=f"q{degree}_delta_energy_variance_by_restriction.png",
        )
        plot_delta_metric_by_restriction_for_degree(
            results,
            q_dir,
            degree,
            metric="delta_p_ground",
            xlabel=r"$\Delta P_{ground}$",
            filename=f"q{degree}_delta_pground_by_restriction.png",
        )
        plot_delta_metric_by_restriction_for_degree(
            results,
            q_dir,
            degree,
            metric="delta_expected_feasible_energy",
            xlabel=r"$\Delta E_{feas}$",
            filename=f"q{degree}_delta_expected_feasible_energy_by_restriction.png",
        )
        plot_delta_metric_by_restriction_for_degree(
            results,
            q_dir,
            degree,
            metric="delta_feasible_energy_variance",
            xlabel=r"$\Delta \mathrm{Var}(E_{feas})$",
            filename=f"q{degree}_delta_feasible_energy_variance_by_restriction.png",
        )
        plot_pground_scatter_best_for_degree(results, q_dir, degree)
        plot_feasible_energy_scatter_best_for_degree(results, q_dir, degree)
        plot_best_method_delta_energy_by_density_for_degree(results, q_dir, degree)
        plot_best_method_delta_feasible_energy_by_density_for_degree(results, q_dir, degree)
        plot_best_method_delta_pground_by_density_for_degree(results, q_dir, degree)
        plot_best_restriction_counts_for_degree(best_by_degree, q_dir, degree)

    print(f"Rebuilt interpretation tables in {tables_dir}.")
    print(f"Saved overview plots to {overview_dir}.")
    print(f"Saved q-specific plots to {by_degree_dir}.")


if __name__ == "__main__":
    main()
