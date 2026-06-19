from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_formulation2 import best_rho_by_energy, confidence_table, overall_summary_table, summarize_by_group


FORMULATION_DIR = Path(__file__).resolve().parent


# raise a clear error if a required input file is missing.
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_formulation2_experiment.py before making plots."
        )


# save the current matplotlib figure as png.
def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# return a dataframe with finite x and y values only.
def finite_pair_df(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()

    return df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()


# compute padded limits for an identity scatter plot.
def identity_axis_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    low = float(np.min(values))
    high = float(np.max(values))

    if np.isclose(low, high):
        padding = max(1.0, abs(low) * 0.05)
    else:
        padding = 0.05 * (high - low)

    return low - padding, high + padding


# produce sorted rho groups while keeping rho=0 first.
def sorted_rho_groups(df: pd.DataFrame):
    groups = []

    for rho, group in df.groupby("rho"):
        groups.append((float(rho), group))

    return sorted(groups, key=lambda item: item[0])


# plot the mean expected-energy trajectory for the baseline and all rho values.
def plot_mean_energy_trajectory(history_df: pd.DataFrame, out_dir: str | Path) -> None:
    mean_df = history_df.groupby(["method", "rho", "step"], dropna=False, as_index=False)["expected_energy"].mean()

    if mean_df.empty:
        return

    plt.figure(figsize=(8, 5))

    baseline = mean_df[mean_df["method"] == "baseline"]
    if not baseline.empty:
        plt.plot(
            baseline["step"],
            baseline["expected_energy"],
            label="baseline",
            linewidth=2.4,
        )

    guided = mean_df[mean_df["method"] != "baseline"]
    for rho, group in sorted_rho_groups(guided):
        plt.plot(
            group["step"],
            group["expected_energy"],
            label=rf"warm-guided, $\rho={rho:g}$",
            linewidth=1.6,
        )

    plt.xlabel("annealing step")
    plt.ylabel("mean expected energy")
    plt.title("Mean energy trajectory")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / "formulation2_mean_energy_trajectory.png")


# plot the mean guiding schedule eta by rho.
def plot_eta_schedules(schedule_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "eta" not in schedule_df.columns:
        return

    guided = schedule_df[schedule_df["method"] != "baseline"]

    if guided.empty:
        return

    mean_df = guided.groupby(["rho", "step"], as_index=False)["eta"].mean()

    plt.figure(figsize=(8, 5))
    for rho, group in sorted_rho_groups(mean_df):
        plt.plot(group["step"], group["eta"], label=rf"$\rho={rho:g}$", linewidth=1.8)

    plt.xlabel("annealing step")
    plt.ylabel(r"guiding coefficient $\eta$")
    plt.title("Warm-guiding schedules")
    plt.legend(fontsize=9)
    save_figure(Path(out_dir) / "formulation2_eta_schedules.png")


# plot paired metric differences by rho.
def plot_delta_metric_by_rho(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns or "rho" not in results_df.columns:
        return

    data = []
    labels = []

    for rho, group in sorted_rho_groups(results_df):
        values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size:
            data.append(values)
            labels.append(f"{rho:g}")

    if not data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"guiding strength $\rho$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by guiding strength")
    save_figure(Path(out_dir) / filename)


# plot paired metric differences by graph size for the best rho.
def plot_delta_metric_by_n_for_best_rho(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns or "n" not in results_df.columns:
        return

    best_rho = best_rho_by_energy(results_df)
    subset = results_df[np.isclose(results_df["rho"], best_rho)]

    if subset.empty:
        return

    data = []
    labels = []

    for key, group in subset.groupby("n"):
        values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size:
            data.append(values)
            labels.append(str(key))

    if not data:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("graph size n")
    plt.ylabel(ylabel)
    plt.title(rf"{ylabel} by graph size, best $\rho={best_rho:g}$")
    save_figure(Path(out_dir) / filename)


# plot paired metric differences by graph density for the best rho.
def plot_delta_metric_by_p_for_best_rho(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns or "p" not in results_df.columns:
        return

    best_rho = best_rho_by_energy(results_df)
    subset = results_df[np.isclose(results_df["rho"], best_rho)]

    if subset.empty:
        return

    data = []
    labels = []

    for key, group in subset.groupby("p"):
        values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size:
            data.append(values)
            labels.append(str(key))

    if not data:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("graph density p")
    plt.ylabel(ylabel)
    plt.title(rf"{ylabel} by graph density, best $\rho={best_rho:g}$")
    save_figure(Path(out_dir) / filename)


# compare baseline and warm-guided ground-state probabilities for the best rho.
def plot_pground_scatter_best_rho(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    best_rho = best_rho_by_energy(results_df)
    subset = results_df[np.isclose(results_df["rho"], best_rho)]

    pairs = finite_pair_df(
        subset,
        "base_ground_state_probability",
        "guided_ground_state_probability",
    )

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_ground_state_probability"],
        pairs["guided_ground_state_probability"],
        alpha=0.7,
    )

    high = max(
        float(pairs["base_ground_state_probability"].max()),
        float(pairs["guided_ground_state_probability"].max()),
        1e-12,
    )
    plt.plot([0, high], [0, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline ground-state probability")
    plt.ylabel("warm-guided ground-state probability")
    plt.title(rf"Ground-state probability comparison, $\rho={best_rho:g}$")
    save_figure(Path(out_dir) / "formulation2_ground_state_probability_scatter_best_rho.png")


# compare baseline and warm-guided expected feasible energies for the best rho.
def plot_expected_feasible_energy_scatter_best_rho(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    best_rho = best_rho_by_energy(results_df)
    subset = results_df[np.isclose(results_df["rho"], best_rho)]

    pairs = finite_pair_df(
        subset,
        "base_expected_feasible_energy",
        "guided_expected_feasible_energy",
    )

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_expected_feasible_energy"],
        pairs["guided_expected_feasible_energy"],
        alpha=0.7,
    )

    low, high = identity_axis_limits(pairs.to_numpy().ravel())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline expected feasible energy")
    plt.ylabel("warm-guided expected feasible energy")
    plt.title(rf"Expected feasible energy comparison, $\rho={best_rho:g}$")
    save_figure(Path(out_dir) / "formulation2_expected_feasible_energy_scatter_best_rho.png")


# plot mean paired improvements by rho and density.
def plot_density_heatmap(results_df: pd.DataFrame, out_dir: str | Path, metric: str, filename: str) -> None:
    if metric not in results_df.columns:
        return

    pivot = results_df.pivot_table(index="p", columns="rho", values=metric, aggfunc="mean")

    if pivot.empty:
        return

    plt.figure(figsize=(8, 5))
    image = plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(image, label=f"mean {metric}")
    plt.xticks(np.arange(len(pivot.columns)), [f"{rho:g}" for rho in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), [str(p) for p in pivot.index])
    plt.xlabel(r"guiding strength $\rho$")
    plt.ylabel("graph density p")
    plt.title(f"Mean {metric} by density and rho")
    save_figure(Path(out_dir) / filename)


# plot the distribution of warm-start and signed guiding entries.
def plot_guiding_vector_distribution(warm_vectors_df: pd.DataFrame, out_dir: str | Path) -> None:
    for column, xlabel, filename in [
        ("c_i", r"warm-start value $c_i^*$", "formulation2_c_vector_distribution.png"),
        ("q_i", r"signed guiding value $q_i$", "formulation2_q_vector_distribution.png"),
    ]:
        if column not in warm_vectors_df.columns:
            continue

        values = warm_vectors_df[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size == 0:
            continue

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=30)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.title(f"Distribution of {column}")
        save_figure(Path(out_dir) / filename)


# plot the runtime components.
def plot_runtime_components(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    if results_df.empty:
        return

    first_per_instance = results_df.drop_duplicates("instance_id")

    data = []
    labels = []

    for column, label, source_df in [
        ("base_simulation_time", "baseline simulation", first_per_instance),
        ("sdp_time", "SDP warm start", first_per_instance),
    ]:
        if column not in source_df.columns:
            continue

        values = source_df[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(label)

    for rho, group in sorted_rho_groups(results_df):
        if "guided_simulation_time" not in group.columns:
            continue

        values = group["guided_simulation_time"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if values.size:
            data.append(values)
            labels.append(rf"guided $\rho={rho:g}$")

    if not data:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("time in seconds")
    plt.title("Runtime components")
    save_figure(Path(out_dir) / "formulation2_runtime_components.png")


# rebuild tables and create all formulation 2 plots.
def main() -> None:
    results_dir = FORMULATION_DIR / "results"
    tables_dir = results_dir / "tables"
    raw_dir = results_dir / "raw"
    plots_dir = results_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "formulation2_per_instance_results.csv"
    histories_path = raw_dir / "formulation2_energy_histories.csv"
    schedules_path = raw_dir / "formulation2_schedules.csv"
    warm_vectors_path = raw_dir / "formulation2_warm_guiding_vectors.csv"

    for path in [results_path, histories_path, schedules_path, warm_vectors_path]:
        require_file(path)

    results = pd.read_csv(results_path)
    histories = pd.read_csv(histories_path)
    schedules = pd.read_csv(schedules_path)
    warm_vectors = pd.read_csv(warm_vectors_path)

    summaries = [
        summarize_by_group(results.assign(overall="all"), ["overall", "rho"]),
        summarize_by_group(results, ["rho"]),
        summarize_by_group(results, ["rho", "n"]),
        summarize_by_group(results, ["rho", "p"]),
        summarize_by_group(results, ["rho", "n", "p"]),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation2_grouped_summary.csv",
        index=False,
    )

    overall_summary_table(results).to_csv(
        tables_dir / "formulation2_overall_summary.csv",
        index=False,
    )

    confidence_table(
        results,
        [
            "delta_energy",
            "delta_energy_variance",
            "delta_p_ground",
            "delta_expected_feasible_energy",
            "delta_feasible_energy_variance",
            "delta_p_feas",
        ],
        group_cols=["rho"],
    ).to_csv(tables_dir / "formulation2_bootstrap_ci.csv", index=False)

    plot_mean_energy_trajectory(histories, plots_dir)
    plot_eta_schedules(schedules, plots_dir)

    plot_delta_metric_by_rho(results, plots_dir, "delta_energy", r"$\Delta E$", "formulation2_delta_energy_by_rho.png")
    plot_delta_metric_by_rho(results, plots_dir, "delta_p_ground", r"$\Delta P_{ground}$", "formulation2_delta_pground_by_rho.png")
    plot_delta_metric_by_rho(results, plots_dir, "delta_expected_feasible_energy", r"$\Delta E_{feas}$", "formulation2_delta_expected_feasible_energy_by_rho.png")
    plot_delta_metric_by_rho(results, plots_dir, "delta_p_feas", r"$\Delta P_{feas}$", "formulation2_delta_pfeas_by_rho.png")

    plot_delta_metric_by_n_for_best_rho(results, plots_dir, "delta_energy", r"$\Delta E$", "formulation2_delta_energy_by_n_best_rho.png")
    plot_delta_metric_by_p_for_best_rho(results, plots_dir, "delta_energy", r"$\Delta E$", "formulation2_delta_energy_by_p_best_rho.png")
    plot_delta_metric_by_n_for_best_rho(results, plots_dir, "delta_p_ground", r"$\Delta P_{ground}$", "formulation2_delta_pground_by_n_best_rho.png")
    plot_delta_metric_by_p_for_best_rho(results, plots_dir, "delta_p_ground", r"$\Delta P_{ground}$", "formulation2_delta_pground_by_p_best_rho.png")

    plot_pground_scatter_best_rho(results, plots_dir)
    plot_expected_feasible_energy_scatter_best_rho(results, plots_dir)
    plot_density_heatmap(results, plots_dir, "delta_energy", "formulation2_delta_energy_density_rho_heatmap.png")
    plot_density_heatmap(results, plots_dir, "delta_p_ground", "formulation2_delta_pground_density_rho_heatmap.png")
    plot_guiding_vector_distribution(warm_vectors, plots_dir)
    plot_runtime_components(results, plots_dir)

    print(f"Rebuilt interpretation tables in {tables_dir}.")
    print(f"Saved plots to {plots_dir}.")


if __name__ == "__main__":
    main()
