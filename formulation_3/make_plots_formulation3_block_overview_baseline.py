from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_formulation3 import confidence_table, overall_summary_table, summarize_by_group


FORMULATION_DIR = Path(__file__).resolve().parent


# raise a clear error if a required input file is missing.
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_formulation3_experiment.py before making plots."
        )


# save the current matplotlib figure as png.
def save_figure(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# convert a method name to a readable plot label.
def method_label(method: str) -> str:
    if str(method) == "baseline":
        return "baseline"
    if str(method).startswith("block_mu_"):
        return rf"$\mu={str(method).split('_')[-1]}$"
    return str(method)


# return sorted integer mu values.
def available_mu_values(df: pd.DataFrame) -> list[int]:
    if "mu" not in df.columns:
        return []
    values = df["mu"].dropna().to_numpy(dtype=int)
    return sorted({int(v) for v in values})


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


# plot a metric as boxplots by a grouping column.
def plot_metric_by_group(
    df: pd.DataFrame,
    out_dir: str | Path,
    group_col: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if metric not in df.columns or group_col not in df.columns:
        return

    groups = []
    for key, group in df.groupby(group_col, dropna=False):
        values = group[metric].dropna().to_numpy(dtype=float)
        if values.size:
            groups.append((key, values))

    if not groups:
        return

    # stable ordering for numeric labels.
    try:
        groups = sorted(groups, key=lambda item: float(item[0]))
    except Exception:
        groups = sorted(groups, key=lambda item: str(item[0]))

    labels = [str(key) for key, _ in groups]
    values = [value for _, value in groups]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=labels)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.title(title)
    save_figure(Path(out_dir) / filename)


# plot a metric as a single bar chart of means by group.
def plot_mean_bar_by_group(
    df: pd.DataFrame,
    out_dir: str | Path,
    group_col: str,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    if metric not in df.columns or group_col not in df.columns:
        return

    mean_df = df.groupby(group_col, as_index=False)[metric].mean().dropna()
    if mean_df.empty:
        return

    try:
        mean_df = mean_df.sort_values(group_col, key=lambda s: s.astype(float))
    except Exception:
        mean_df = mean_df.sort_values(group_col)

    plt.figure(figsize=(8, 5))
    plt.bar(mean_df[group_col].astype(str), mean_df[metric].astype(float))
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.title(title)
    save_figure(Path(out_dir) / filename)


# plot the mean expected-energy trajectory for selected methods.
def plot_mean_energy_trajectory(
    history_df: pd.DataFrame,
    out_dir: str | Path,
    filename: str,
    title: str,
    methods: set[str] | None = None,
) -> None:
    df = history_df.copy()
    if methods is not None:
        df = df[df["method"].isin(methods)]

    mean_df = df.groupby(["method", "step"], as_index=False)["expected_energy"].mean()
    if mean_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, group in mean_df.groupby("method"):
        linewidth = 2.4 if method == "baseline" else 1.9
        plt.plot(group["step"], group["expected_energy"], label=method_label(method), linewidth=linewidth)

    plt.xlabel("annealing step")
    plt.ylabel("mean expected energy")
    plt.title(title)
    plt.legend(fontsize=9)
    save_figure(Path(out_dir) / filename)


# plot the mean expected energy under the formulation-specific problem Hamiltonian.
def plot_block_problem_energy_trajectory(
    history_df: pd.DataFrame,
    out_dir: str | Path,
    filename: str,
    title: str,
    methods: set[str] | None = None,
) -> None:
    if "block_problem_expected_energy" not in history_df.columns:
        return

    plot_frames = []

    block_df = history_df.dropna(subset=["block_problem_expected_energy"]).copy()
    if methods is not None:
        block_df = block_df[block_df["method"].isin(methods)]

    if not block_df.empty:
        block_mean_df = block_df.groupby(["method", "step"], as_index=False)["block_problem_expected_energy"].mean()
        block_mean_df = block_mean_df.rename(columns={"block_problem_expected_energy": "plot_energy"})
        plot_frames.append(block_mean_df[["method", "step", "plot_energy"]])

    if "expected_energy" in history_df.columns:
        baseline_df = history_df[history_df["method"] == "baseline"].dropna(subset=["expected_energy"]).copy()
        if methods is not None:
            baseline_df = baseline_df[baseline_df["method"].isin(methods)]

        if not baseline_df.empty:
            baseline_mean_df = baseline_df.groupby(["method", "step"], as_index=False)["expected_energy"].mean()
            baseline_mean_df = baseline_mean_df.rename(columns={"expected_energy": "plot_energy"})
            plot_frames.append(baseline_mean_df[["method", "step", "plot_energy"]])

    if not plot_frames:
        return

    mean_df = pd.concat(plot_frames, ignore_index=True)

    plt.figure(figsize=(8, 5))
    methods_order = sorted(mean_df["method"].unique(), key=lambda name: (name != "baseline", str(name)))
    for method in methods_order:
        group = mean_df[mean_df["method"] == method]
        linewidth = 2.4 if method == "baseline" else 1.9
        plt.plot(group["step"], group["plot_energy"], label=method_label(method), linewidth=linewidth)

    plt.xlabel("annealing step")
    plt.ylabel("mean expected energy")
    plt.title(title)
    plt.legend(fontsize=9)
    save_figure(Path(out_dir) / filename)


# plot block-valid probability trajectories.
def plot_block_valid_trajectory(
    history_df: pd.DataFrame,
    out_dir: str | Path,
    filename: str,
    title: str,
    methods: set[str] | None = None,
) -> None:
    if "block_valid_probability" not in history_df.columns:
        return

    df = history_df.dropna(subset=["block_valid_probability"]).copy()
    if methods is not None:
        df = df[df["method"].isin(methods)]

    mean_df = df.groupby(["method", "step"], as_index=False)["block_valid_probability"].mean()
    if mean_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for method, group in mean_df.groupby("method"):
        if method == "baseline":
            continue
        plt.plot(group["step"], group["block_valid_probability"], label=method_label(method), linewidth=1.9)

    plt.xlabel("annealing step")
    plt.ylabel("mean block-valid probability")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=9)
    save_figure(Path(out_dir) / filename)


# scatter comparison with identity line for one mu.
def plot_identity_scatter_for_mu(
    results_df: pd.DataFrame,
    out_dir: str | Path,
    mu: int,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    sub = results_df[results_df["mu"] == int(mu)] if "mu" in results_df.columns else results_df
    pair_df = finite_pair_df(sub, x_col, y_col)
    if pair_df.empty:
        return

    values = pair_df[[x_col, y_col]].to_numpy(dtype=float).reshape(-1)
    low, high = identity_axis_limits(values)

    plt.figure(figsize=(6, 6))
    plt.scatter(pair_df[x_col], pair_df[y_col], alpha=0.75, s=22)
    plt.plot([low, high], [low, high], linestyle="--", linewidth=1.0)
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_figure(Path(out_dir) / filename)


# paired runtime comparison for a single mu.
def plot_runtime_for_mu(results_df: pd.DataFrame, out_dir: str | Path, mu: int) -> None:
    sub = results_df[results_df["mu"] == int(mu)].copy()
    if sub.empty or "base_simulation_time" not in sub.columns or "block_simulation_time" not in sub.columns:
        return

    values = [
        sub["base_simulation_time"].dropna().to_numpy(dtype=float),
        sub["block_simulation_time"].dropna().to_numpy(dtype=float),
    ]
    if not any(value.size for value in values):
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=["baseline", rf"$\mu={mu}$"])
    plt.ylabel("time in seconds")
    plt.title(rf"Runtime comparison for $\mu={mu}$")
    save_figure(Path(out_dir) / "runtime_comparison.png")


# runtime overview by mu.
def plot_runtime_overview(results_df: pd.DataFrame, out_dir: str | Path) -> None:
    if "mu" not in results_df.columns or "block_simulation_time" not in results_df.columns:
        return

    groups = []
    for mu, group in results_df.groupby("mu"):
        values = group["block_simulation_time"].dropna().to_numpy(dtype=float)
        if values.size:
            groups.append((int(mu), values))

    if not groups:
        return

    groups = sorted(groups, key=lambda item: item[0])
    labels = [str(mu) for mu, _ in groups]
    values = [value for _, value in groups]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=labels)
    plt.xlabel(r"$\mu$")
    plt.ylabel("time in seconds")
    plt.title(r"Block-driver simulation time by $\mu$")
    save_figure(Path(out_dir) / "runtime_by_mu.png")


# plot block-size distribution for one mu using block-assignment rows.
def plot_block_size_distribution(block_df: pd.DataFrame, out_dir: str | Path, mu: int) -> None:
    if block_df.empty or "mu" not in block_df.columns or "block_size" not in block_df.columns:
        return

    sub = block_df[block_df["mu"] == int(mu)]
    if sub.empty:
        return

    block_sizes = sub.drop_duplicates(["instance_id", "block"])["block_size"].dropna().to_numpy(dtype=int)
    if block_sizes.size == 0:
        return

    bins = np.arange(0.5, int(mu) + 1.6, 1.0)
    plt.figure(figsize=(7, 5))
    plt.hist(block_sizes, bins=bins, rwidth=0.85)
    plt.xticks(range(1, int(mu) + 1))
    plt.xlabel("block size")
    plt.ylabel("count")
    plt.title(rf"Block-size distribution for $\mu={mu}$")
    save_figure(Path(out_dir) / "block_size_distribution.png")


# create all overview plots.
def make_overview_plots(results_df: pd.DataFrame, histories_df: pd.DataFrame, overview_dir: Path) -> None:
    overview_dir.mkdir(parents=True, exist_ok=True)

    overview_methods = {"baseline"} | {
        f"block_mu_{int(mu)}" for mu in available_mu_values(results_df)
    }

    plot_mean_energy_trajectory(
        histories_df,
        overview_dir,
        "mean_energy_trajectory_all_mu.png",
        r"Mean energy trajectory by $\mu$",
        methods=overview_methods,
    )
    plot_block_valid_trajectory(
        histories_df,
        overview_dir,
        "block_valid_probability_trajectory_all_mu.png",
        r"Block-valid probability trajectory by $\mu$",
    )
    plot_block_problem_energy_trajectory(
        histories_df,
        overview_dir,
        "block_problem_energy_trajectory_all_mu.png",
        r"Mean block-problem energy trajectory by $\mu$",
        methods=overview_methods,
    )

    plot_metric_by_group(results_df, overview_dir, "mu", "delta_energy", r"$\Delta E$", r"Final-energy difference by $\mu$", "delta_energy_by_mu.png")
    plot_metric_by_group(results_df, overview_dir, "mu", "delta_p_ground", r"$\Delta P_{ground}$", r"Ground-state probability difference by $\mu$", "delta_pground_by_mu.png")
    plot_metric_by_group(results_df, overview_dir, "mu", "delta_expected_feasible_energy", r"$\Delta E_{feas}$", r"Expected feasible-energy difference by $\mu$", "delta_expected_feasible_energy_by_mu.png")
    plot_metric_by_group(results_df, overview_dir, "mu", "delta_p_feas", r"$\Delta P_{feas}$", r"Feasibility probability difference by $\mu$", "delta_pfeas_by_mu.png")

    plot_runtime_overview(results_df, overview_dir)
    plot_mean_bar_by_group(results_df, overview_dir, "mu", "num_blocks", "number of blocks", r"Mean number of blocks by $\mu$", "num_blocks_by_mu.png")
    plot_mean_bar_by_group(results_df, overview_dir, "mu", "mean_block_size", "mean block size", r"Mean block size by $\mu$", "mean_block_size_by_mu.png")
    plot_mean_bar_by_group(results_df, overview_dir, "mu", "block_final_block_valid_probability", "final block-valid probability", r"Final block-valid probability by $\mu$", "block_valid_probability_by_mu.png")


# create per-mu plots.
def make_mu_plots(
    results_df: pd.DataFrame,
    histories_df: pd.DataFrame,
    block_df: pd.DataFrame,
    base_mu_dir: Path,
    mu: int,
) -> None:
    mu_dir = base_mu_dir / f"mu_{int(mu)}"
    mu_dir.mkdir(parents=True, exist_ok=True)

    method = f"block_mu_{int(mu)}"
    methods = {"baseline", method}
    sub = results_df[results_df["mu"] == int(mu)].copy()

    plot_mean_energy_trajectory(
        histories_df,
        mu_dir,
        "mean_energy_trajectory.png",
        rf"Mean energy trajectory for $\mu={mu}$",
        methods=methods,
    )
    plot_block_valid_trajectory(
        histories_df,
        mu_dir,
        "block_valid_probability_trajectory.png",
        rf"Block-valid probability trajectory for $\mu={mu}$",
        methods={method},
    )
    plot_block_problem_energy_trajectory(
        histories_df,
        mu_dir,
        "block_problem_energy_trajectory.png",
        rf"Block-problem energy trajectory for $\mu={mu}$",
        methods={method},
    )

    plot_identity_scatter_for_mu(
        sub,
        mu_dir,
        int(mu),
        "base_ground_state_probability",
        "block_ground_state_probability",
        "baseline ground-state probability",
        rf"block-driver ground-state probability ($\mu={mu}$)",
        rf"Ground-state probability comparison for $\mu={mu}$",
        "ground_probability_scatter.png",
    )
    plot_identity_scatter_for_mu(
        sub,
        mu_dir,
        int(mu),
        "base_final_expected_energy",
        "block_final_expected_energy",
        "baseline final expected energy",
        rf"block-driver final expected energy ($\mu={mu}$)",
        rf"Final expected-energy comparison for $\mu={mu}$",
        "final_energy_scatter.png",
    )
    plot_identity_scatter_for_mu(
        sub,
        mu_dir,
        int(mu),
        "base_expected_feasible_energy",
        "block_expected_feasible_energy",
        "baseline expected feasible energy",
        rf"block-driver expected feasible energy ($\mu={mu}$)",
        rf"Expected feasible-energy comparison for $\mu={mu}$",
        "expected_feasible_energy_scatter.png",
    )

    plot_metric_by_group(sub, mu_dir, "n", "delta_energy", r"$\Delta E$", rf"Final-energy difference by graph size for $\mu={mu}$", "delta_energy_by_size.png")
    plot_metric_by_group(sub, mu_dir, "p", "delta_energy", r"$\Delta E$", rf"Final-energy difference by density for $\mu={mu}$", "delta_energy_by_density.png")
    plot_metric_by_group(sub, mu_dir, "n", "delta_p_ground", r"$\Delta P_{ground}$", rf"Ground-state probability difference by graph size for $\mu={mu}$", "delta_pground_by_size.png")
    plot_metric_by_group(sub, mu_dir, "p", "delta_p_ground", r"$\Delta P_{ground}$", rf"Ground-state probability difference by density for $\mu={mu}$", "delta_pground_by_density.png")
    plot_metric_by_group(sub, mu_dir, "n", "delta_expected_feasible_energy", r"$\Delta E_{feas}$", rf"Expected feasible-energy difference by graph size for $\mu={mu}$", "delta_expected_feasible_energy_by_size.png")
    plot_metric_by_group(sub, mu_dir, "p", "delta_expected_feasible_energy", r"$\Delta E_{feas}$", rf"Expected feasible-energy difference by density for $\mu={mu}$", "delta_expected_feasible_energy_by_density.png")

    plot_runtime_for_mu(sub, mu_dir, int(mu))
    plot_mean_bar_by_group(sub, mu_dir, "n", "num_blocks", "number of blocks", rf"Number of blocks by graph size for $\mu={mu}$", "num_blocks_by_size.png")
    plot_mean_bar_by_group(sub, mu_dir, "p", "num_blocks", "number of blocks", rf"Number of blocks by density for $\mu={mu}$", "num_blocks_by_density.png")
    plot_block_size_distribution(block_df, mu_dir, int(mu))


# write summary tables and create plots.
def main() -> None:
    results_dir = FORMULATION_DIR / "results"
    tables_dir = results_dir / "tables"
    raw_dir = results_dir / "raw"
    plots_dir = results_dir / "plots"
    overview_dir = plots_dir / "overview"
    by_mu_dir = plots_dir / "by_mu"
    tables_by_mu_dir = tables_dir / "by_mu"

    results_path = tables_dir / "formulation3_per_instance_results.csv"
    histories_path = raw_dir / "formulation3_energy_histories.csv"
    block_assignments_path = raw_dir / "formulation3_block_assignments.csv"

    require_file(results_path)
    require_file(histories_path)

    results_df = pd.read_csv(results_path)
    histories_df = pd.read_csv(histories_path)
    block_df = pd.read_csv(block_assignments_path) if block_assignments_path.exists() else pd.DataFrame()

    plots_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    by_mu_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables_by_mu_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_by_group(results_df.assign(overall="all"), ["overall"]),
        summarize_by_group(results_df, ["mu"]),
        summarize_by_group(results_df, ["n"]),
        summarize_by_group(results_df, ["p"]),
        summarize_by_group(results_df, ["mu", "n"]),
        summarize_by_group(results_df, ["mu", "p"]),
        summarize_by_group(results_df, ["mu", "n", "p"]),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation3_grouped_summary.csv",
        index=False,
    )

    overall_summary_table(results_df).to_csv(
        tables_dir / "formulation3_overall_summary.csv",
        index=False,
    )

    confidence_table(
        results_df,
        [
            "delta_energy",
            "delta_energy_variance",
            "delta_p_ground",
            "delta_expected_feasible_energy",
            "delta_feasible_energy_variance",
            "delta_p_feas",
        ],
    ).to_csv(tables_dir / "formulation3_bootstrap_ci.csv", index=False)

    make_overview_plots(results_df, histories_df, overview_dir)

    for mu in available_mu_values(results_df):
        make_mu_plots(results_df, histories_df, block_df, by_mu_dir, mu)

        mu_sub = results_df[results_df["mu"] == int(mu)].copy()
        mu_table_dir = tables_by_mu_dir / f"mu_{int(mu)}"
        mu_table_dir.mkdir(parents=True, exist_ok=True)
        summarize_by_group(mu_sub.assign(mu_value=int(mu)), ["mu_value"]).to_csv(mu_table_dir / "summary.csv", index=False)
        summarize_by_group(mu_sub, ["n"]).to_csv(mu_table_dir / "summary_by_size.csv", index=False)
        summarize_by_group(mu_sub, ["p"]).to_csv(mu_table_dir / "summary_by_density.csv", index=False)

    print(f"Saved overview plots to {overview_dir}.")
    print(f"Saved per-mu plots to {by_mu_dir}.")
    print(f"Saved summary tables to {tables_dir}.")


if __name__ == "__main__":
    main()
