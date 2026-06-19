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

from analysis_formulation1 import confidence_table, overall_summary_table, summarize_by_group
from warm_start_annealing import (
    clip_warm_start_vector,
    simulate_warm_start_annealing,
    solve_sdp_warm_start,
)
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

    try:
        import cvxpy as cp

        versions["cvxpy"] = cp.__version__
        versions["cvxpy_installed_solvers"] = cp.installed_solvers()
    except Exception as exc:
        versions["cvxpy"] = f"not available: {exc}"

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
        meta = parse_instance_name(path)
        n_ok = n_values is None or int(meta.get("n", -1)) in n_values
        p_ok = p_set is None or round(float(meta.get("p", -1.0)), 6) in p_set

        if n_ok and p_ok:
            selected.append(path)

    return selected


# store one energy-history row per annealing grid point.
def append_energy_history(
    rows: list[dict],
    result: dict,
    instance_base: dict,
    method: str,
) -> None:
    for step, energy in enumerate(result["energy_history"]):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "step": int(step),
                "expected_energy": float(energy),
            }
        )


# build the baseline schedule table.
def baseline_schedule_rows(instance_base: dict, n_steps: int) -> list[dict]:
    alpha_grid, beta_grid = coefficient_schedule_grid(n_steps)
    rows = []

    rows.append(
        {
            "instance_id": instance_base["instance_id"],
            "n": instance_base["n"],
            "p": instance_base["p"],
            "replicate": instance_base["replicate"],
            "method": "baseline",
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
                "method": "baseline",
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


# run baseline and warm-start annealing on one instance.
def run_single_instance(path: Path, args) -> tuple[dict, list[dict], list[dict], list[dict]]:
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

    t0_sdp = time.perf_counter()
    sdp = solve_sdp_warm_start(
        n=n,
        edges=edges,
        penalty=args.penalty,
        solver=args.solver,
        eps_solver=args.sdp_tol,
        max_iters=args.sdp_max_iters,
        verbose=args.sdp_verbose,
    )
    c = clip_warm_start_vector(sdp["x_value"], args.clip_eps)
    sdp_time = time.perf_counter() - t0_sdp

    t0_warm = time.perf_counter()
    warm = simulate_warm_start_annealing(
        cost_energies=cost_energies,
        c=c,
        T=args.total_time,
        N_steps=args.n_steps,
        angle_scale=args.angle_scale,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        ode_method=args.ode_method,
    )
    warm_time = time.perf_counter() - t0_warm

    warm_metrics = final_state_metrics(
        state=warm["final_state"],
        cost_energies=cost_energies,
        bitstrings=bitstrings,
        edges=edges,
    )

    row = {
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
        "sdp_time": sdp_time,
        "base_simulation_time": base_time,
        "warm_simulation_time": warm_time,
        "sdp_status": sdp["status"],
        "sdp_solver": sdp["solver"],
        "sdp_objective": sdp["objective_value"],
        "sdp_solver_time": sdp["solve_time"],
        "sdp_num_iters": sdp["num_iters"],
        "clip_eps": float(args.clip_eps),
        "warm_c_min": float(np.min(c)),
        "warm_c_max": float(np.max(c)),
        "warm_c_mean": float(np.mean(c)),
        "warm_c_std": float(np.std(c, ddof=1)) if c.size > 1 else 0.0,
        "base_max_norm_deviation": float(np.max(np.abs(base["norm_history"] - 1.0))),
        "base_raw_max_norm_deviation": float(
            np.max(np.abs(base.get("raw_norm_history", base["norm_history"]) - 1.0))
        ),
        "warm_max_norm_deviation": float(np.max(np.abs(warm["norm_history"] - 1.0))),
        "warm_raw_max_norm_deviation": float(
            np.max(np.abs(warm.get("raw_norm_history", warm["norm_history"]) - 1.0))
        ),
        "evolution_method": "continuous_ode_linear_schedule",
        "warm_mixer_type": "warm_start_mixer",
        "ode_method": args.ode_method,
        "ode_rtol": float(args.ode_rtol),
        "ode_atol": float(args.ode_atol),
    }

    add_metrics_with_prefix(row, "base", base_metrics)
    add_metrics_with_prefix(row, "warm", warm_metrics)

    row["delta_energy"] = row["warm_final_expected_energy"] - row["base_final_expected_energy"]
    row["delta_energy_variance"] = row["warm_energy_variance"] - row["base_energy_variance"]
    row["delta_expected_feasible_energy"] = (
        row["warm_expected_feasible_energy"] - row["base_expected_feasible_energy"]
    )
    row["delta_feasible_energy_variance"] = (
        row["warm_feasible_energy_variance"] - row["base_feasible_energy_variance"]
    )
    row["delta_p_ground"] = (
        row["warm_ground_state_probability"] - row["base_ground_state_probability"]
    )
    row["delta_p_feas"] = row["warm_feasibility_probability"] - row["base_feasibility_probability"]

    histories = []
    append_energy_history(histories, base, row, "baseline")
    append_energy_history(histories, warm, row, "warm_start_sdp")

    schedules = baseline_schedule_rows(row, args.n_steps)

    for schedule_row in baseline_schedule_rows(row, args.n_steps):
        warm_row = dict(schedule_row)
        warm_row["method"] = "warm_start_sdp"
        schedules.append(warm_row)

    warm_vector_rows = []
    theta = warm["theta"]

    for qubit, ci in enumerate(c):
        warm_vector_rows.append(
            {
                "instance_id": row["instance_id"],
                "n": row["n"],
                "p": row["p"],
                "replicate": row["replicate"],
                "qubit": int(qubit),
                "c_i": float(ci),
                "theta_i": float(theta[qubit]),
            }
        )

    return row, histories, schedules, warm_vector_rows


# parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and SDP warm-start annealing for Maximum Clique."
    )

    parser.add_argument("--instances-dir", type=Path, default=REPOSITORY_DIR / "ER_instances")
    parser.add_argument("--results-dir", type=Path, default=FORMULATION_DIR / "results")
    parser.add_argument("--n-values", type=int, nargs="*", default=None)
    parser.add_argument("--p-values", type=float, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--ode-method", type=str, default="DOP853")
    parser.add_argument("--ode-rtol", type=float, default=1e-8)
    parser.add_argument("--ode-atol", type=float, default=1e-10)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--sdp-tol", type=float, default=1e-6)
    parser.add_argument("--sdp-max-iters", type=int, default=20000)
    parser.add_argument("--sdp-verbose", action="store_true")
    parser.add_argument("--clip-eps", type=float, default=1e-6)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)

    return parser.parse_args()


# run the full experiment.
def main() -> None:
    args = parse_args()

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
    warm_vectors = []

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}")
        row, history_rows, schedule_rows, warm_vector_rows = run_single_instance(path, args)
        results.append(row)
        histories.extend(history_rows)
        schedules.extend(schedule_rows)
        warm_vectors.extend(warm_vector_rows)

    results_df = pd.DataFrame(results)
    histories_df = pd.DataFrame(histories)
    schedules_df = pd.DataFrame(schedules)
    warm_vectors_df = pd.DataFrame(warm_vectors)

    results_df.to_csv(tables_dir / "formulation1_per_instance_results.csv", index=False)
    histories_df.to_csv(raw_dir / "formulation1_energy_histories.csv", index=False)
    schedules_df.to_csv(raw_dir / "formulation1_schedules.csv", index=False)
    warm_vectors_df.to_csv(raw_dir / "formulation1_warm_start_vectors.csv", index=False)

    summaries = [
        summarize_by_group(results_df.assign(overall="all"), ["overall"]),
        summarize_by_group(results_df, ["n"]),
        summarize_by_group(results_df, ["p"]),
        summarize_by_group(results_df, ["n", "p"]),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation1_grouped_summary.csv",
        index=False,
    )

    overall_summary_table(results_df).to_csv(
        tables_dir / "formulation1_overall_summary.csv",
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
    ).to_csv(tables_dir / "formulation1_bootstrap_ci.csv", index=False)

    metadata = vars(args).copy()
    metadata.update(
        {
            "num_instances_run": len(paths),
            "method": "warm_start_sdp",
            "package_versions": package_versions(),
        }
    )

    with (logs_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print(f"Saved tables to {tables_dir} and raw outputs to {raw_dir}.")


if __name__ == "__main__":
    main()
