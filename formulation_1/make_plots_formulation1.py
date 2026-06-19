from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_formulation1 import confidence_table, overall_summary_table, summarize_by_group


FORMULATION_DIR = Path(__file__).resolve().parent


# raise a clear error if a required input file is missing.
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_formulation1_experiment.py before making plots."
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


# plot the mean expected-energy trajectory.
def plot_mean_energy_trajectory(history_df: pd.DataFrame, out_dir: str | Path) -> None:
    mean_df = history_df.groupby(["method", "step"], as_index=False)["expected_energy"].mean()

    if mean_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for method, group in mean_df.groupby("method"):
        label = "warm-start" if method == "warm_start_sdp" else "baseline"
        linewidth = 2.4 if method == "baseline" else 1.8
        plt.plot(group["step"], group["expected_energy"], label=label, linewidth=linewidth)

    plt.xlabel("annealing step")
    plt.ylabel("mean expected energy")
    plt.title("Mean energy trajectory")
    plt.legend(fontsize=9)
    save_figure(Path(out_dir) / "formulation1_mean_energy_trajectory.png")


# plot paired metric differences by graph size.
def plot_delta_metric_by_n(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns:
        return

    groups = list(results_df.groupby("n"))

    if not groups:
        return

    data = []
    labels = []

    for key, group in groups:
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
    plt.title(f"{ylabel} by graph size")
    save_figure(Path(out_dir) / filename)


# plot paired metric differences by graph density.
def plot_delta_metric_by_p(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns:
        return

    groups = list(results_df.groupby("p"))

    if not groups:
        return

    data = []
    labels = []

    for key, group in groups:
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
    plt.title(f"{ylabel} by graph density")
    save_figure(Path(out_dir) / filename)


# compare baseline and warm-start ground-state probabilities.
def plot_pground_scatter(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    pairs = finite_pair_df(
        results_df,
        "base_ground_state_probability",
        "warm_ground_state_probability",
    )

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_ground_state_probability"],
        pairs["warm_ground_state_probability"],
        alpha=0.7,
    )

    high = max(
        float(pairs["base_ground_state_probability"].max()),
        float(pairs["warm_ground_state_probability"].max()),
        1e-12,
    )
    plt.plot([0, high], [0, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline ground-state probability")
    plt.ylabel("warm-start ground-state probability")
    plt.title("Ground-state probability comparison")
    save_figure(Path(out_dir) / "formulation1_ground_state_probability_scatter.png")


# compare baseline and warm-start expected feasible energies.
def plot_expected_feasible_energy_scatter(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    pairs = finite_pair_df(
        results_df,
        "base_expected_feasible_energy",
        "warm_expected_feasible_energy",
    )

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_expected_feasible_energy"],
        pairs["warm_expected_feasible_energy"],
        alpha=0.7,
    )

    low, high = identity_axis_limits(pairs.to_numpy().ravel())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline expected feasible energy")
    plt.ylabel("warm-start expected feasible energy")
    plt.title("Expected feasible energy comparison")
    save_figure(Path(out_dir) / "formulation1_expected_feasible_energy_scatter.png")


# plot the SDP and simulation runtime components.
def plot_runtime_components(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    columns = ["base_simulation_time", "sdp_time", "warm_simulation_time"]
    labels = ["baseline simulation", "SDP warm start", "warm-start simulation"]

    data = []
    plot_labels = []

    for column, label in zip(columns, labels):
        if column not in results_df.columns:
            continue

        values = results_df[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size:
            data.append(values)
            plot_labels.append(label)

    if not data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=plot_labels)
    plt.ylabel("time in seconds")
    plt.title("Runtime components")
    save_figure(Path(out_dir) / "formulation1_runtime_components.png")


# plot the distribution of warm-start vector entries.
def plot_warm_start_vector_distribution(warm_vectors_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "c_i" not in warm_vectors_df.columns:
        return

    values = warm_vectors_df["c_i"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if values.size == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=30)
    plt.xlabel(r"warm-start value $c_i$")
    plt.ylabel("count")
    plt.title("Distribution of warm-start vector entries")
    save_figure(Path(out_dir) / "formulation1_warm_start_vector_distribution.png")


# rebuild tables and create all formulation 1 plots.
def main() -> None:
    results_dir = FORMULATION_DIR / "results"
    tables_dir = results_dir / "tables"
    raw_dir = results_dir / "raw"
    plots_dir = results_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / "formulation1_per_instance_results.csv"
    histories_path = raw_dir / "formulation1_energy_histories.csv"
    warm_vectors_path = raw_dir / "formulation1_warm_start_vectors.csv"

    for path in [results_path, histories_path, warm_vectors_path]:
        require_file(path)

    results = pd.read_csv(results_path)
    histories = pd.read_csv(histories_path)
    warm_vectors = pd.read_csv(warm_vectors_path)

    summaries = [
        summarize_by_group(results.assign(overall="all"), ["overall"]),
        summarize_by_group(results, ["n"]),
        summarize_by_group(results, ["p"]),
        summarize_by_group(results, ["n", "p"]),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation1_grouped_summary.csv",
        index=False,
    )

    overall_summary_table(results).to_csv(
        tables_dir / "formulation1_overall_summary.csv",
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
    ).to_csv(tables_dir / "formulation1_bootstrap_ci.csv", index=False)

    plot_mean_energy_trajectory(histories, plots_dir)
    plot_delta_metric_by_n(results, plots_dir, "delta_energy", r"$\Delta E$", "formulation1_delta_energy_by_n.png")
    plot_delta_metric_by_p(results, plots_dir, "delta_energy", r"$\Delta E$", "formulation1_delta_energy_by_p.png")
    plot_delta_metric_by_n(
        results,
        plots_dir,
        "delta_p_ground",
        r"$\Delta P_{ground}$",
        "formulation1_delta_pground_by_n.png",
    )
    plot_delta_metric_by_p(
        results,
        plots_dir,
        "delta_p_ground",
        r"$\Delta P_{ground}$",
        "formulation1_delta_pground_by_p.png",
    )
    plot_delta_metric_by_n(
        results,
        plots_dir,
        "delta_expected_feasible_energy",
        r"$\Delta E_{feas}$",
        "formulation1_delta_expected_feasible_energy_by_n.png",
    )
    plot_delta_metric_by_p(
        results,
        plots_dir,
        "delta_expected_feasible_energy",
        r"$\Delta E_{feas}$",
        "formulation1_delta_expected_feasible_energy_by_p.png",
    )
    plot_pground_scatter(results, plots_dir)
    plot_expected_feasible_energy_scatter(results, plots_dir)
    plot_runtime_components(results, plots_dir)
    plot_warm_start_vector_distribution(warm_vectors, plots_dir)

    print(f"Rebuilt interpretation tables in {tables_dir}.")
    print(f"Saved plots to {plots_dir}.")


if __name__ == "__main__":
    main()
