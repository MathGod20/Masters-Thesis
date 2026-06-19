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

from analysis_formulation2 import confidence_table, overall_summary_table, summarize_by_group
from warm_guiding_annealing import (
    clip_warm_start_vector,
    signed_guiding_coefficients,
    simulate_warm_guiding_annealing,
    solve_sdp_warm_start,
    warm_guiding_schedule_grid,
)
from base_setup.graph_io import parse_instance_name, read_edgelist
from base_setup.metrics import feasibility_mask, final_state_metrics
from base_setup.qubo import bitstring_to_str, build_cost_energies, build_max_clique_qubo
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


# return a stable text label for one rho value.
def rho_method_label(rho: float) -> str:
    label = f"{float(rho):.4g}".replace("-", "m").replace(".", "p")
    return f"warm_guiding_rho_{label}"


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
    rho: float | None,
) -> None:
    for step, energy in enumerate(result["energy_history"]):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "rho": np.nan if rho is None else float(rho),
                "step": int(step),
                "expected_energy": float(energy),
            }
        )


# build schedule rows for the baseline.
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
            "rho": np.nan,
            "step": 0,
            "alpha": float(alpha_grid[0]),
            "alpha_left": np.nan,
            "alpha_right": float(alpha_grid[0]),
            "beta": float(beta_grid[0]),
            "beta_left": np.nan,
            "beta_right": float(beta_grid[0]),
            "eta": 0.0,
            "eta_left": np.nan,
            "eta_right": 0.0,
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
                "rho": np.nan,
                "step": int(step),
                "alpha": float(alpha_grid[step]),
                "alpha_left": float(alpha_grid[step - 1]),
                "alpha_right": float(alpha_grid[step]),
                "beta": float(beta_grid[step]),
                "beta_left": float(beta_grid[step - 1]),
                "beta_right": float(beta_grid[step]),
                "eta": 0.0,
                "eta_left": 0.0,
                "eta_right": 0.0,
            }
        )

    return rows


# build schedule rows for Formulation 2.
def guided_schedule_rows(instance_base: dict, n_steps: int, rho: float) -> list[dict]:
    alpha_grid, beta_grid = coefficient_schedule_grid(n_steps)
    eta_grid = warm_guiding_schedule_grid(n_steps, rho)
    method = rho_method_label(rho)
    rows = []

    rows.append(
        {
            "instance_id": instance_base["instance_id"],
            "n": instance_base["n"],
            "p": instance_base["p"],
            "replicate": instance_base["replicate"],
            "method": method,
            "rho": float(rho),
            "step": 0,
            "alpha": float(alpha_grid[0]),
            "alpha_left": np.nan,
            "alpha_right": float(alpha_grid[0]),
            "beta": float(beta_grid[0]),
            "beta_left": np.nan,
            "beta_right": float(beta_grid[0]),
            "eta": float(eta_grid[0]),
            "eta_left": np.nan,
            "eta_right": float(eta_grid[0]),
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
                "rho": float(rho),
                "step": int(step),
                "alpha": float(alpha_grid[step]),
                "alpha_left": float(alpha_grid[step - 1]),
                "alpha_right": float(alpha_grid[step]),
                "beta": float(beta_grid[step]),
                "beta_left": float(beta_grid[step - 1]),
                "beta_right": float(beta_grid[step]),
                "eta": float(eta_grid[step]),
                "eta_left": float(eta_grid[step - 1]),
                "eta_right": float(eta_grid[step]),
            }
        )

    return rows


# store final probability rows. This can be large, so it can be disabled from the CLI.
def append_probability_rows(
    rows: list[dict],
    state: np.ndarray,
    instance_base: dict,
    method: str,
    rho: float | None,
    cost_energies: np.ndarray,
    bitstrings: list[tuple[int, ...]],
    feasible: np.ndarray,
    probability_threshold: float,
) -> None:
    probabilities = np.abs(state) ** 2
    ground_energy = float(np.min(cost_energies))
    threshold = float(probability_threshold)

    for index, probability in enumerate(probabilities):
        if float(probability) < threshold:
            continue

        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "rho": np.nan if rho is None else float(rho),
                "basis_index": int(index),
                "bitstring": bitstring_to_str(bitstrings[index]),
                "probability": float(probability),
                "qubo_energy": float(cost_energies[index]),
                "is_feasible_clique": bool(feasible[index]),
                "is_ground_state": bool(np.isclose(cost_energies[index], ground_energy)),
            }
        )


# run baseline and all warm-guiding rho values on one instance.
def run_single_instance(
    path: Path,
    args,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    metadata = parse_instance_name(path)

    t0_common = time.perf_counter()
    n, edges = read_edgelist(path)
    qubo = build_max_clique_qubo(n, edges, reward=args.reward, penalty=args.penalty)
    cost_energies, bitstrings = build_cost_energies(n, qubo)
    feasible = feasibility_mask(bitstrings, edges)
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
    q = signed_guiding_coefficients(c)
    sdp_time = time.perf_counter() - t0_sdp

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
        "sdp_time": sdp_time,
        "base_simulation_time": base_time,
        "sdp_status": sdp["status"],
        "sdp_solver": sdp["solver"],
        "sdp_objective": sdp["objective_value"],
        "sdp_solver_time": sdp["solve_time"],
        "sdp_num_iters": sdp["num_iters"],
        "clip_eps": float(args.clip_eps),
        "c_min": float(np.min(c)),
        "c_max": float(np.max(c)),
        "c_mean": float(np.mean(c)),
        "c_std": float(np.std(c, ddof=1)) if c.size > 1 else 0.0,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "q_mean": float(np.mean(q)),
        "q_std": float(np.std(q, ddof=1)) if q.size > 1 else 0.0,
        "base_max_norm_deviation": float(np.max(np.abs(base["norm_history"] - 1.0))),
        "base_raw_max_norm_deviation": float(
            np.max(np.abs(base.get("raw_norm_history", base["norm_history"]) - 1.0))
        ),
        "evolution_method": "continuous_ode_linear_schedule",
        "guiding_type": "warm_guiding_diagonal_field",
        "ode_method": args.ode_method,
        "ode_rtol": float(args.ode_rtol),
        "ode_atol": float(args.ode_atol),
    }

    histories = []
    schedules = []
    warm_vectors = []
    probabilities = []
    rows = []

    append_energy_history(histories, base, instance_base, "baseline", None)
    schedules.extend(baseline_schedule_rows(instance_base, args.n_steps))

    if not args.skip_final_probabilities:
        append_probability_rows(
            probabilities,
            base["final_state"],
            instance_base,
            "baseline",
            None,
            cost_energies,
            bitstrings,
            feasible,
            args.probability_threshold,
        )

    for qubit, ci in enumerate(c):
        warm_vectors.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "qubit": int(qubit),
                "c_i": float(ci),
                "q_i": float(q[qubit]),
            }
        )

    for rho in args.rho_values:
        t0_guided = time.perf_counter()
        guided = simulate_warm_guiding_annealing(
            cost_energies=cost_energies,
            q=q,
            T=args.total_time,
            N_steps=args.n_steps,
            rho=float(rho),
            angle_scale=args.angle_scale,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_method=args.ode_method,
        )
        guided_time = time.perf_counter() - t0_guided

        guided_metrics = final_state_metrics(
            state=guided["final_state"],
            cost_energies=cost_energies,
            bitstrings=bitstrings,
            edges=edges,
        )

        row = dict(instance_base)
        row.update(
            {
                "rho": float(rho),
                "method": rho_method_label(float(rho)),
                "eta_max": float(np.max(guided["eta_grid"])),
                "guided_simulation_time": guided_time,
                "guided_max_norm_deviation": float(np.max(np.abs(guided["norm_history"] - 1.0))),
                "guided_raw_max_norm_deviation": float(
                    np.max(np.abs(guided.get("raw_norm_history", guided["norm_history"]) - 1.0))
                ),
            }
        )

        add_metrics_with_prefix(row, "base", base_metrics)
        add_metrics_with_prefix(row, "guided", guided_metrics)

        row["delta_energy"] = row["guided_final_expected_energy"] - row["base_final_expected_energy"]
        row["delta_energy_variance"] = row["guided_energy_variance"] - row["base_energy_variance"]
        row["delta_expected_feasible_energy"] = (
            row["guided_expected_feasible_energy"] - row["base_expected_feasible_energy"]
        )
        row["delta_feasible_energy_variance"] = (
            row["guided_feasible_energy_variance"] - row["base_feasible_energy_variance"]
        )
        row["delta_p_ground"] = (
            row["guided_ground_state_probability"] - row["base_ground_state_probability"]
        )
        row["delta_p_feas"] = (
            row["guided_feasibility_probability"] - row["base_feasibility_probability"]
        )

        rows.append(row)
        append_energy_history(histories, guided, instance_base, rho_method_label(float(rho)), float(rho))
        schedules.extend(guided_schedule_rows(instance_base, args.n_steps, float(rho)))

        if not args.skip_final_probabilities:
            append_probability_rows(
                probabilities,
                guided["final_state"],
                instance_base,
                rho_method_label(float(rho)),
                float(rho),
                cost_energies,
                bitstrings,
                feasible,
                args.probability_threshold,
            )

    return rows, histories, schedules, warm_vectors, probabilities


# parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and warm-guided annealing for Maximum Clique."
    )

    parser.add_argument("--instances-dir", type=Path, default=REPOSITORY_DIR / "ER_instances")
    parser.add_argument("--results-dir", type=Path, default=FORMULATION_DIR / "results")
    parser.add_argument("--n-values", type=int, nargs="*", default=None)
    parser.add_argument("--p-values", type=float, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--rho-values", type=float, nargs="+", default=[0.0, 0.25, 0.50, 1.00, 2.00])
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
    parser.add_argument("--skip-final-probabilities", action="store_true")
    parser.add_argument("--probability-threshold", type=float, default=0.0)

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
    probabilities = []

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}")
        row_list, history_rows, schedule_rows, warm_vector_rows, probability_rows = run_single_instance(path, args)
        results.extend(row_list)
        histories.extend(history_rows)
        schedules.extend(schedule_rows)
        warm_vectors.extend(warm_vector_rows)
        probabilities.extend(probability_rows)

    results_df = pd.DataFrame(results)
    histories_df = pd.DataFrame(histories)
    schedules_df = pd.DataFrame(schedules)
    warm_vectors_df = pd.DataFrame(warm_vectors)

    results_df.to_csv(tables_dir / "formulation2_per_instance_results.csv", index=False)
    histories_df.to_csv(raw_dir / "formulation2_energy_histories.csv", index=False)
    schedules_df.to_csv(raw_dir / "formulation2_schedules.csv", index=False)
    warm_vectors_df.to_csv(raw_dir / "formulation2_warm_guiding_vectors.csv", index=False)

    if not args.skip_final_probabilities:
        probabilities_df = pd.DataFrame(probabilities)
        probabilities_df.to_csv(raw_dir / "formulation2_final_probabilities.csv", index=False)

    summaries = [
        summarize_by_group(results_df.assign(overall="all"), ["overall", "rho"]),
        summarize_by_group(results_df, ["rho"]),
        summarize_by_group(results_df, ["rho", "n"]),
        summarize_by_group(results_df, ["rho", "p"]),
        summarize_by_group(results_df, ["rho", "n", "p"]),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation2_grouped_summary.csv",
        index=False,
    )

    overall_summary_table(results_df).to_csv(
        tables_dir / "formulation2_overall_summary.csv",
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
        group_cols=["rho"],
    ).to_csv(tables_dir / "formulation2_bootstrap_ci.csv", index=False)

    metadata = vars(args).copy()
    metadata.update(
        {
            "num_instances_run": len(paths),
            "method": "warm_guiding",
            "package_versions": package_versions(),
        }
    )

    with (logs_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print(f"Saved tables to {tables_dir} and raw outputs to {raw_dir}.")


if __name__ == "__main__":
    main()
