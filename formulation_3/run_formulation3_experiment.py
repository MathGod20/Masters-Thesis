from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FORMULATION_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = FORMULATION_DIR.parent

if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from analysis_formulation3 import confidence_table, overall_summary_table, summarize_by_group
from block_mu_annealing import simulate_block_mu_annealing
from base_setup.graph_io import parse_instance_name, read_edgelist
from base_setup.metrics import final_state_metrics
from base_setup.qubo import build_cost_energies, build_max_clique_qubo
from base_setup.statevector import coefficient_schedule_grid, simulate_baseline


# collect package and system versions.
def package_versions() -> dict:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception as exc:
        versions["scipy"] = f"not available: {exc}"

    return versions


# add metrics to a row using a prefix.
def add_metrics_with_prefix(row: dict, prefix: str, metrics: dict) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


# select instance files from the benchmark folder.
def collect_instance_paths(
    instances_dir: Path,
    n_values: list[int] | None,
    p_values: list[float] | None,
) -> list[Path]:
    paths = sorted(instances_dir.glob("*.edgelist"))
    selected = []
    p_set = None if p_values is None else {round(float(p), 6) for p in p_values}

    for path in paths:
        metadata = parse_instance_name(path)
        n_ok = n_values is None or int(metadata.get("n", -1)) in n_values
        p_ok = p_set is None or round(float(metadata.get("p", -1.0)), 6) in p_set

        if n_ok and p_ok:
            selected.append(path)

    return selected


# store one energy-history row per annealing grid point.
def append_energy_history(
    rows: list[dict],
    result: dict,
    instance_base: dict,
    method: str,
    mu: int | None,
) -> None:
    block_history = result.get("block_problem_energy_history")
    block_valid_history = result.get("block_valid_probability_history")

    for step, energy in enumerate(result["energy_history"]):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "mu": np.nan if mu is None else int(mu),
                "step": int(step),
                "expected_energy": float(energy),
                "block_problem_expected_energy": (
                    float(block_history[step]) if block_history is not None else np.nan
                ),
                "block_valid_probability": (
                    float(block_valid_history[step]) if block_valid_history is not None else np.nan
                ),
            }
        )


# build the fixed endpoint schedule table.
def fixed_schedule_rows(instance_base: dict, n_steps: int, method: str, mu: int | None) -> list[dict]:
    alpha_grid, beta_grid = coefficient_schedule_grid(n_steps)
    rows = []

    rows.append(
        {
            "instance_id": instance_base["instance_id"],
            "n": instance_base["n"],
            "p": instance_base["p"],
            "replicate": instance_base["replicate"],
            "method": method,
            "mu": np.nan if mu is None else int(mu),
            "step": 0,
            "alpha": float(alpha_grid[0]),
            "alpha_left": np.nan,
            "alpha_right": float(alpha_grid[0]),
            "beta": float(beta_grid[0]),
            "beta_left": np.nan,
            "beta_right": float(beta_grid[0]),
        }
    )

    for step in range(1, int(n_steps) + 1):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "mu": np.nan if mu is None else int(mu),
                "step": int(step),
                "alpha": float(alpha_grid[step]),
                "alpha_left": float(alpha_grid[step - 1]),
                "alpha_right": float(alpha_grid[step]),
                "beta": float(beta_grid[step]),
                "beta_left": float(beta_grid[step - 1]),
                "beta_right": float(beta_grid[step]),
            }
        )

    return rows


# create rows describing the greedy independent blocks.
def block_assignment_rows(instance_base: dict, result: dict) -> list[dict]:
    rows = []
    blocks = result["blocks"]
    mu = int(result["mu"])

    for block_index, block in enumerate(blocks):
        for vertex in block:
            rows.append(
                {
                    "instance_id": instance_base["instance_id"],
                    "n": instance_base["n"],
                    "p": instance_base["p"],
                    "replicate": instance_base["replicate"],
                    "mu": mu,
                    "vertex": int(vertex),
                    "block": int(block_index),
                    "block_size": int(len(block)),
                }
            )

    return rows


# run baseline once and block-mu variants on one instance.
def run_single_instance(path: Path, args) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    metadata = parse_instance_name(path)

    t0_common = time.perf_counter()
    n, edges = read_edgelist(path)
    qubo = build_max_clique_qubo(n, edges, reward=args.reward, penalty=args.penalty)
    cost_energies, bitstrings = build_cost_energies(n, qubo)
    common_time = time.perf_counter() - t0_common

    t0_base = time.perf_counter()
    base = simulate_baseline(
        cost_energies=cost_energies,
        n=n,
        T=args.total_time,
        N_steps=args.n_steps,
        angle_scale=args.angle_scale,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_method=args.ode_method,
    )
    base_time = time.perf_counter() - t0_base

    base_metrics = final_state_metrics(
        state=base["final_state"],
        cost_energies=cost_energies,
        bitstrings=bitstrings,
        edges=edges,
    )

    instance_base = {
        "instance_id": metadata.get("instance_id", path.stem),
        "path": str(path),
        "n": int(metadata.get("n", n)),
        "p": float(metadata.get("p", float("nan"))),
        "replicate": int(metadata.get("replicate", -1)),
        "num_edges": int(len(edges)),
        "density_realized": 2 * len(edges) / (n * (n - 1)) if n > 1 else 0.0,
        "T": float(args.total_time),
        "N_steps": int(args.n_steps),
        "dt": float(args.total_time) / int(args.n_steps),
        "angle_scale": float(args.angle_scale),
        "common_time": common_time,
        "base_simulation_time": base_time,
        "base_max_norm_deviation": float(np.max(np.abs(base["norm_history"] - 1.0))),
        "base_raw_max_norm_deviation": float(
            np.max(np.abs(base.get("raw_norm_history", base["norm_history"]) - 1.0))
        ),
        "evolution_method": "continuous_ode_linear_schedule",
        "fixed_alpha_schedule": True,
        "fixed_beta_schedule": True,
        "ode_method": args.ode_method,
        "ode_rtol": float(args.ode_rtol),
        "ode_atol": float(args.ode_atol),
    }
    add_metrics_with_prefix(instance_base, "base", base_metrics)

    results = []
    histories = []
    schedules = []
    block_rows = []

    append_energy_history(histories, base, instance_base, "baseline", None)
    schedules.extend(fixed_schedule_rows(instance_base, args.n_steps, "baseline", None))

    for mu in args.mu_values:
        t0_block = time.perf_counter()
        block = simulate_block_mu_annealing(
            n=n,
            edges=edges,
            evaluation_energies=cost_energies,
            T=args.total_time,
            N_steps=args.n_steps,
            mu=int(mu),
            reward=args.reward,
            penalty=args.penalty,
            block_order=args.block_order,
            angle_scale=args.angle_scale,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_method=args.ode_method,
        )
        block_time = time.perf_counter() - t0_block

        block_metrics = final_state_metrics(
            state=block["final_state"],
            cost_energies=cost_energies,
            bitstrings=bitstrings,
            edges=edges,
        )

        block_sizes = np.asarray(block["block_sizes"], dtype=float)

        row = dict(instance_base)
        row.update(
            {
                "method": f"block_mu_{int(mu)}",
                "mu": int(mu),
                "block_order": args.block_order,
                "num_blocks": int(block["num_blocks"]),
                "min_block_size": int(np.min(block_sizes)) if block_sizes.size else 0,
                "max_block_size": int(np.max(block_sizes)) if block_sizes.size else 0,
                "mean_block_size": float(np.mean(block_sizes)) if block_sizes.size else float("nan"),
                "block_driver_ground_energy": float(block["block_driver_ground_energy"]),
                "block_final_block_valid_probability": float(block["block_valid_probability_history"][-1]),
                "block_min_block_valid_probability": float(np.min(block["block_valid_probability_history"])),
                "block_simulation_time": block_time,
                "block_construction_time": float(block.get("block_construction_time", np.nan)),
                "driver_precompute_time": float(block.get("driver_precompute_time", np.nan)),
                "block_problem_energy_time": float(block.get("block_problem_energy_time", np.nan)),
                "block_initial_state_time": float(block.get("block_initial_state_time", np.nan)),
                "block_setup_time": float(block.get("block_setup_time", np.nan)),
                "block_total_recorded_time": float(
                    block.get("block_setup_time", 0.0) + block_time
                ),
                "block_max_norm_deviation": float(np.max(np.abs(block["norm_history"] - 1.0))),
                "block_raw_max_norm_deviation": float(
                    np.max(np.abs(block.get("raw_norm_history", block["norm_history"]) - 1.0))
                ),
                "driver_type": "block_mu_driver",
                "initial_state_type": "block_mu_driver_closed_form_ground_state",
                "problem_type": "block_mu_reduced_max_clique_penalty",
            }
        )

        add_metrics_with_prefix(row, "block", block_metrics)

        row["delta_energy"] = row["block_final_expected_energy"] - row["base_final_expected_energy"]
        row["delta_energy_variance"] = row["block_energy_variance"] - row["base_energy_variance"]
        row["delta_expected_feasible_energy"] = (
            row["block_expected_feasible_energy"] - row["base_expected_feasible_energy"]
        )
        row["delta_feasible_energy_variance"] = (
            row["block_feasible_energy_variance"] - row["base_feasible_energy_variance"]
        )
        row["delta_p_ground"] = row["block_ground_state_probability"] - row["base_ground_state_probability"]
        row["delta_p_feas"] = row["block_feasibility_probability"] - row["base_feasibility_probability"]

        results.append(row)
        append_energy_history(histories, block, row, f"block_mu_{int(mu)}", int(mu))
        schedules.extend(fixed_schedule_rows(row, args.n_steps, f"block_mu_{int(mu)}", int(mu)))
        block_rows.extend(block_assignment_rows(row, block))

    return results, histories, schedules, block_rows


# parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and block-mu driver annealing for Maximum Clique."
    )

    parser.add_argument("--instances-dir", type=Path, default=REPOSITORY_DIR / "ER_instances")
    parser.add_argument("--results-dir", type=Path, default=FORMULATION_DIR / "results")
    parser.add_argument("--n-values", type=int, nargs="*", default=None)
    parser.add_argument("--p-values", type=float, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mu-values", type=int, nargs="*", default=[2, 3, 5, 999])
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--ode-method", type=str, default="DOP853")
    parser.add_argument("--ode-rtol", type=float, default=1e-8)
    parser.add_argument("--ode-atol", type=float, default=1e-10)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument(
        "--block-order",
        type=str,
        default="largest_degree_first",
        choices=["largest_degree_first", "natural"],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)

    return parser.parse_args()


# run the full experiment.
def main() -> None:
    args = parse_args()

    if not args.mu_values:
        raise ValueError("At least one mu value must be provided.")

    if any(int(mu) < 1 for mu in args.mu_values):
        raise ValueError("All mu values must be at least 1.")

    tables_dir = args.results_dir / "tables"
    raw_dir = args.results_dir / "raw"
    logs_dir = args.results_dir / "logs"

    for folder in [tables_dir, raw_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    paths = collect_instance_paths(args.instances_dir, args.n_values, args.p_values)

    if args.limit is not None:
        paths = paths[: args.limit]

    if not paths:
        raise FileNotFoundError(f"No .edgelist files found in {args.instances_dir}.")

    results = []
    histories = []
    schedules = []
    block_assignments = []

    start_time = time.perf_counter()

    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path.name}")
        row_list, history_rows, schedule_rows, block_rows = run_single_instance(path, args)
        results.extend(row_list)
        histories.extend(history_rows)
        schedules.extend(schedule_rows)
        block_assignments.extend(block_rows)

    total_runtime = time.perf_counter() - start_time

    results_df = pd.DataFrame(results)
    histories_df = pd.DataFrame(histories)
    schedules_df = pd.DataFrame(schedules)
    block_df = pd.DataFrame(block_assignments)

    results_df.to_csv(tables_dir / "formulation3_per_instance_results.csv", index=False)
    histories_df.to_csv(raw_dir / "formulation3_energy_histories.csv", index=False)
    schedules_df.to_csv(raw_dir / "formulation3_schedules.csv", index=False)
    block_df.to_csv(raw_dir / "formulation3_block_assignments.csv", index=False)

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

    runtime_columns = [
        "block_construction_time",
        "driver_precompute_time",
        "block_problem_energy_time",
        "block_initial_state_time",
        "block_setup_time",
        "block_simulation_time",
        "block_total_recorded_time",
    ]
    (
        results_df
        .groupby("mu", as_index=False)[runtime_columns]
        .agg(["mean", "median", "std"])
    ).to_csv(tables_dir / "formulation3_runtime_summary.csv")

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
        args.bootstrap_resamples,
    ).to_csv(tables_dir / "formulation3_bootstrap_ci.csv", index=False)

    metadata = vars(args).copy()
    metadata.update(
        {
            "num_instances_run": len(paths),
            "num_rows": int(len(results_df)),
            "total_runtime_seconds": total_runtime,
            "method": "block_mu_driver",
            "package_versions": package_versions(),
        }
    )

    with (logs_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print(f"Saved tables to {tables_dir} and raw outputs to {raw_dir}.")


if __name__ == "__main__":
    main()
