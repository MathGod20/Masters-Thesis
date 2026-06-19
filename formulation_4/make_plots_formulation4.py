from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_formulation4 import best_by_instance_table, rho_summary_table


FORMULATION_DIR = Path(__file__).resolve().parent


# raise a clear error if a required input file is missing.
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_formulation4_experiment.py before making plots."
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


# choose the rho value with best mean final expected energy.
def best_rho_by_mean_energy(results_df: pd.DataFrame) -> float:
    summary = rho_summary_table(results_df)

    if summary.empty:
        return float("nan")

    best = summary.sort_values(
        ["mean_delta_energy", "mean_delta_p_ground"],
        ascending=[True, False],
    ).iloc[0]

    return float(best["rho"])


# create a label for one rho value.
def rho_label(rho: float) -> str:
    return rf"$\rho={float(rho):g}$"


# plot the mean expected-energy trajectory.
def plot_mean_energy_trajectory(history_df: pd.DataFrame, out_dir: str | Path) -> None:
    mean_df = history_df.groupby(["method", "rho", "method_id", "step"], dropna=False, as_index=False)[
        "expected_energy"
    ].mean()

    if mean_df.empty:
        return

    plt.figure(figsize=(8, 5))

    baseline = mean_df[mean_df["method"] == "baseline"]
    if not baseline.empty:
        base_group = baseline.groupby("step", as_index=False)["expected_energy"].mean()
        plt.plot(base_group["step"], base_group["expected_energy"], label="baseline", linewidth=2.4)

    assisted = mean_df[mean_df["method"] == "qaoa_mixer_assisted"]
    for rho, group in assisted.groupby("rho"):
        plt.plot(group["step"], group["expected_energy"], label=rho_label(rho), linewidth=1.5)

    plt.xlabel("annealing step")
    plt.ylabel("mean expected energy")
    plt.title("Mean energy trajectory")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / "formulation4_mean_energy_trajectory.png")


# plot the shared problem and driver schedules together with the auxiliary schedules.
def plot_mean_schedules(schedule_df: pd.DataFrame, out_dir: str | Path) -> None:
    if schedule_df.empty:
        return

    mean_df = schedule_df.groupby(["method", "rho", "method_id", "step"], dropna=False, as_index=False)[
        ["alpha", "beta", "delta"]
    ].mean()

    assisted = mean_df[mean_df["method"] == "qaoa_mixer_assisted"]

    if assisted.empty:
        return

    plt.figure(figsize=(8, 5))

    shared = assisted.groupby("step", as_index=False)[["alpha", "beta"]].mean()
    plt.plot(shared["step"], shared["alpha"], label="alpha", linewidth=2.2)
    plt.plot(shared["step"], shared["beta"], label="beta", linewidth=2.2)

    for rho, group in assisted.groupby("rho"):
        plt.plot(group["step"], group["delta"], label=rf"delta, {rho_label(rho)}", linewidth=1.4)

    plt.xlabel("annealing step")
    plt.ylabel("schedule value")
    plt.title("Problem, driver, and auxiliary schedules")
    plt.legend(fontsize=8)
    save_figure(Path(out_dir) / "formulation4_mean_schedules.png")


# plot paired metric differences by auxiliary strength.
def plot_delta_metric_by_rho(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns:
        return

    groups = list(results_df.sort_values("rho").groupby("rho"))

    if not groups:
        return

    data = []
    labels = []

    for key, group in groups:
        values = group[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

        if values.size:
            data.append(values)
            labels.append(rho_label(key))

    if not data:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("auxiliary mixer strength")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by auxiliary strength")
    plt.xticks(rotation=20)
    save_figure(Path(out_dir) / filename)


# plot paired metric differences by graph size for the best rho value.
def plot_delta_metric_by_n(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    best_rho: float,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns or not np.isfinite(best_rho):
        return

    df = results_df[np.isclose(results_df["rho"], best_rho)].copy()
    groups = list(df.groupby("n"))

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
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("graph size n")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by graph size, {rho_label(best_rho)}")
    save_figure(Path(out_dir) / filename)


# plot paired metric differences by graph density for the best rho value.
def plot_delta_metric_by_p(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    best_rho: float,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    if metric not in results_df.columns or not np.isfinite(best_rho):
        return

    df = results_df[np.isclose(results_df["rho"], best_rho)].copy()
    groups = list(df.groupby("p"))

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
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("graph density p")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by graph density, {rho_label(best_rho)}")
    save_figure(Path(out_dir) / filename)


# compare baseline and assisted ground-state probabilities for the best rho value.
def plot_pground_scatter(results_df: pd.DataFrame, out_dir: str | Path, best_rho: float) -> None:
    if not np.isfinite(best_rho):
        return

    df = results_df[np.isclose(results_df["rho"], best_rho)].copy()
    pairs = finite_pair_df(df, "base_ground_state_probability", "assisted_ground_state_probability")

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_ground_state_probability"],
        pairs["assisted_ground_state_probability"],
        alpha=0.7,
    )

    high = max(
        float(pairs["base_ground_state_probability"].max()),
        float(pairs["assisted_ground_state_probability"].max()),
        1e-12,
    )
    plt.plot([0, high], [0, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline ground-state probability")
    plt.ylabel("assisted ground-state probability")
    plt.title(f"Ground-state probability comparison, {rho_label(best_rho)}")
    save_figure(Path(out_dir) / "formulation4_ground_state_probability_scatter.png")


# compare baseline and assisted expected feasible energies for the best rho value.
def plot_expected_feasible_energy_scatter(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    best_rho: float,
) -> None:
    if not np.isfinite(best_rho):
        return

    df = results_df[np.isclose(results_df["rho"], best_rho)].copy()
    pairs = finite_pair_df(df, "base_expected_feasible_energy", "assisted_expected_feasible_energy")

    if pairs.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(
        pairs["base_expected_feasible_energy"],
        pairs["assisted_expected_feasible_energy"],
        alpha=0.7,
    )

    low, high = identity_axis_limits(pairs.to_numpy().ravel())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1)
    plt.xlabel("baseline expected feasible energy")
    plt.ylabel("assisted expected feasible energy")
    plt.title(f"Expected feasible energy comparison, {rho_label(best_rho)}")
    save_figure(Path(out_dir) / "formulation4_expected_feasible_energy_scatter.png")


# plot the runtime components.
def plot_runtime_components(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    columns = ["base_simulation_time", "assisted_simulation_time"]
    labels = ["baseline simulation", "assisted simulation"]

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

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, tick_labels=plot_labels)
    plt.ylabel("time in seconds")
    plt.title("Runtime components")
    save_figure(Path(out_dir) / "formulation4_runtime_components.png")


# plot mean runtime against mean final-energy improvement.
def plot_runtime_vs_improvement(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    summary = rho_summary_table(results_df)

    if summary.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(summary["mean_assisted_simulation_time"], summary["mean_delta_energy"], alpha=0.8)

    for _, row in summary.iterrows():
        plt.text(
            row["mean_assisted_simulation_time"],
            row["mean_delta_energy"],
            rho_label(row["rho"]),
            fontsize=8,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("mean assisted simulation time")
    plt.ylabel("mean final-energy difference")
    plt.title("Runtime versus improvement")
    save_figure(Path(out_dir) / "formulation4_runtime_vs_improvement.png")


# plot the number of times each rho value is best after observing the instance.
def plot_best_rho_counts(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    best_df = best_by_instance_table(results_df)

    if best_df.empty:
        return

    counts = best_df["best_rho"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar([rho_label(rho) for rho in counts.index], counts.values)
    plt.xlabel("best auxiliary strength")
    plt.ylabel("number of instances")
    plt.title("Best auxiliary strength by instance")
    plt.xticks(rotation=20)
    save_figure(Path(out_dir) / "formulation4_best_rho_counts.png")


# plot auxiliary mixer structural diagnostics.
def plot_auxiliary_mixer_diagnostics(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    columns = [
        "mean_non_neighbor_count",
        "mean_allowed_auxiliary_basis_flips_per_qubit",
    ]

    if not all(column in results_df.columns for column in columns):
        return

    df = results_df.drop_duplicates("instance_id").copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)

    if df.empty:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df["mean_non_neighbor_count"], df["mean_allowed_auxiliary_basis_flips_per_qubit"], alpha=0.7)
    plt.xlabel("mean non-neighbour count")
    plt.ylabel("mean allowed auxiliary flips per qubit")
    plt.title("Auxiliary mixer structure")
    save_figure(Path(out_dir) / "formulation4_auxiliary_mixer_structure.png")


# create all plots and summary tables.
def main() -> None:
    results_dir = FORMULATION_DIR / "results"
    figures_dir = FORMULATION_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "formulation4_results.csv"
    history_path = results_dir / "energy_histories.csv"
    schedule_path = results_dir / "schedules.csv"

    require_file(results_path)
    require_file(history_path)
    require_file(schedule_path)

    results_df = pd.read_csv(results_path)
    history_df = pd.read_csv(history_path)
    schedule_df = pd.read_csv(schedule_path)

    best_rho = best_rho_by_mean_energy(results_df)

    plot_mean_energy_trajectory(history_df, figures_dir)
    plot_mean_schedules(schedule_df, figures_dir)

    plot_delta_metric_by_rho(
        results_df,
        figures_dir,
        "delta_energy",
        "delta E",
        "formulation4_delta_energy_by_rho.png",
    )
    plot_delta_metric_by_rho(
        results_df,
        figures_dir,
        "delta_p_ground",
        "delta P_ground",
        "formulation4_delta_pground_by_rho.png",
    )
    plot_delta_metric_by_rho(
        results_df,
        figures_dir,
        "delta_expected_feasible_energy",
        "delta E_feas",
        "formulation4_delta_expected_feasible_energy_by_rho.png",
    )
    plot_delta_metric_by_rho(
        results_df,
        figures_dir,
        "delta_p_feas",
        "delta P_feas",
        "formulation4_delta_pfeas_by_rho.png",
    )

    plot_pground_scatter(results_df, figures_dir, best_rho)
    plot_expected_feasible_energy_scatter(results_df, figures_dir, best_rho)

    plot_delta_metric_by_n(
        results_df,
        figures_dir,
        best_rho,
        "delta_energy",
        "delta E",
        "formulation4_delta_energy_by_n_best_rho.png",
    )
    plot_delta_metric_by_p(
        results_df,
        figures_dir,
        best_rho,
        "delta_energy",
        "delta E",
        "formulation4_delta_energy_by_p_best_rho.png",
    )
    plot_delta_metric_by_n(
        results_df,
        figures_dir,
        best_rho,
        "delta_p_ground",
        "delta P_ground",
        "formulation4_delta_pground_by_n_best_rho.png",
    )
    plot_delta_metric_by_p(
        results_df,
        figures_dir,
        best_rho,
        "delta_p_ground",
        "delta P_ground",
        "formulation4_delta_pground_by_p_best_rho.png",
    )

    plot_runtime_components(results_df, figures_dir)
    plot_runtime_vs_improvement(results_df, figures_dir)
    plot_best_rho_counts(results_df, figures_dir)
    plot_auxiliary_mixer_diagnostics(results_df, figures_dir)

    rho_summary_table(results_df).to_csv(results_dir / "plot_summary_by_rho.csv", index=False)
    best_by_instance_table(results_df).to_csv(results_dir / "plot_best_by_instance.csv", index=False)

    print(f"Saved plots to {figures_dir}")


if __name__ == "__main__":
    main()
