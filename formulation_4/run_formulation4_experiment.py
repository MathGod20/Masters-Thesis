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

from analysis_formulation4 import (
    best_by_instance_table,
    confidence_table,
    overall_summary_table,
    rho_summary_table,
    summarize_by_group,
)
from qaoa_mixer_assisted_annealing import (
    auxiliary_envelope_grid,
    auxiliary_mixer_diagnostics,
    auxiliary_mixer_specification_rows,
    auxiliary_schedule_grid,
    build_allowed_qaoa_mixer_flip_data,
    formulation4_alpha_beta_schedule_grid,
    qaoa_mixer_spectral_norm,
    simulate_qaoa_mixer_assisted_annealing,
    standard_driver_spectral_norm,
)
from base_setup.graph_io import parse_instance_name, read_edgelist
from base_setup.metrics import final_state_metrics
from base_setup.qubo import build_cost_energies, build_max_clique_qubo, bitstring_to_str
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
    method_id = "baseline" if rho is None else f"rho_{float(rho):g}"

    for step, energy in enumerate(result["energy_history"]):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "method_id": method_id,
                "rho": float(rho) if rho is not None else np.nan,
                "step": int(step),
                "expected_energy": float(energy),
            }
        )


# build the schedule table for one method.
def schedule_rows(
    instance_base: dict,
    n_steps: int,
    method: str,
    rho: float | None,
) -> list[dict]:
    if rho is None:
        alpha_grid, beta_grid = coefficient_schedule_grid(n_steps)
        delta_grid = np.zeros_like(alpha_grid)
        envelope_grid = np.zeros_like(alpha_grid)
        method_id = "baseline"
    else:
        alpha_grid, beta_grid = formulation4_alpha_beta_schedule_grid(n_steps)
        envelope_grid = auxiliary_envelope_grid(n_steps)
        delta_grid = auxiliary_schedule_grid(n_steps, rho)
        method_id = f"rho_{float(rho):g}"

    rows = []

    for step in range(int(n_steps) + 1):
        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "method_id": method_id,
                "rho": float(rho) if rho is not None else np.nan,
                "step": int(step),
                "alpha": float(alpha_grid[step]),
                "alpha_left": float(alpha_grid[step - 1]) if step > 0 else np.nan,
                "alpha_right": float(alpha_grid[step]),
                "beta": float(beta_grid[step]),
                "beta_left": float(beta_grid[step - 1]) if step > 0 else np.nan,
                "beta_right": float(beta_grid[step]),
                "auxiliary_envelope": float(envelope_grid[step]),
                "delta": float(delta_grid[step]),
                "delta_left": float(delta_grid[step - 1]) if step > 0 else np.nan,
                "delta_right": float(delta_grid[step]),
            }
        )

    return rows


# save final probabilities in the full bitstring space.
def append_probability_rows(
    rows: list[dict],
    state: np.ndarray,
    bitstrings,
    cost_energies: np.ndarray,
    instance_base: dict,
    method: str,
    rho: float | None,
    probability_tol: float,
) -> None:
    probabilities = np.abs(state) ** 2
    method_id = "baseline" if rho is None else f"rho_{float(rho):g}"

    for index, probability in enumerate(probabilities):
        if probability < probability_tol:
            continue

        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": method,
                "method_id": method_id,
                "rho": float(rho) if rho is not None else np.nan,
                "basis_index": int(index),
                "bitstring": bitstring_to_str(bitstrings[index]),
                "probability": float(probability),
                "baseline_qubo_energy": float(cost_energies[index]),
            }
        )


# run baseline and QAOA-mixer-assisted annealing on one instance.
def run_single_instance(
    path: Path,
    args,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    metadata = parse_instance_name(path)

    t0_common = time.perf_counter()
    n, edges = read_edgelist(path)
    qubo = build_max_clique_qubo(n, edges, reward=args.reward, penalty=args.penalty)
    cost_energies, bitstrings = build_cost_energies(n, qubo)
    qaoa_flip_data = build_allowed_qaoa_mixer_flip_data(n, edges)
    common_time = time.perf_counter() - t0_common

    instance_base = {
        "instance_id": metadata.get("instance_id", path.stem),
        "path": str(path),
        "n": int(metadata.get("n", n)),
        "p": float(metadata.get("p", float("nan"))),
        "replicate": int(metadata.get("replicate", -1)),
    }

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

    histories = []
    schedules = []
    probability_rows = []
    result_rows = []

    append_energy_history(histories, base, instance_base, "baseline", None)
    schedules.extend(schedule_rows(instance_base, args.n_steps, "baseline", None))

    if args.save_probabilities:
        append_probability_rows(
            probability_rows,
            base["final_state"],
            bitstrings,
            cost_energies,
            instance_base,
            "baseline",
            None,
            args.probability_tol,
        )

    t0_norms = time.perf_counter()
    standard_norm = standard_driver_spectral_norm(n)
    qaoa_norm = qaoa_mixer_spectral_norm(
        n=n,
        flip_data=qaoa_flip_data,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
    )
    spectral_norm_time = time.perf_counter() - t0_norms
    mixer_stats = auxiliary_mixer_diagnostics(
        n,
        edges,
        qaoa_flip_data,
        standard_driver_norm=standard_norm,
        qaoa_mixer_norm=qaoa_norm,
    )
    mixer_rows = auxiliary_mixer_specification_rows(instance_base, n, edges, qaoa_flip_data)

    for rho in args.rho_values:
        rho = float(rho)
        method_id = f"rho_{rho:g}"

        t0_assisted = time.perf_counter()
        assisted = simulate_qaoa_mixer_assisted_annealing(
            n=n,
            edges=edges,
            cost_energies=cost_energies,
            T=args.total_time,
            N_steps=args.n_steps,
            rho=rho,
            angle_scale=args.angle_scale,
            initial_state=None,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_method=args.ode_method,
            eig_tol=args.eig_tol,
            eig_maxiter=args.eig_maxiter,
            qaoa_flip_data=qaoa_flip_data,
            standard_driver_norm=standard_norm,
            qaoa_mixer_norm=qaoa_norm,
        )
        assisted_time = time.perf_counter() - t0_assisted

        assisted_metrics = final_state_metrics(
            state=assisted["final_state"],
            cost_energies=cost_energies,
            bitstrings=bitstrings,
            edges=edges,
        )

        row = dict(instance_base)
        row.update(
            {
                "method_id": method_id,
                "rho": rho,
                "standard_driver_spectral_norm": float(standard_norm),
                "qaoa_mixer_spectral_norm": float(qaoa_norm),
                "num_edges": int(len(edges)),
                "density_realized": 2 * len(edges) / (n * (n - 1)) if n > 1 else 0.0,
                "T": float(args.total_time),
                "N_steps": int(args.n_steps),
                "dt": float(args.total_time) / int(args.n_steps),
                "angle_scale": float(args.angle_scale),
                "common_time": common_time,
                "spectral_norm_time": spectral_norm_time,
                "base_simulation_time": base_time,
                "assisted_simulation_time": assisted_time,
                "base_max_norm_deviation": float(np.max(np.abs(base["norm_history"] - 1.0))),
                "base_raw_max_norm_deviation": float(
                    np.max(np.abs(base.get("raw_norm_history", base["norm_history"]) - 1.0))
                ),
                "assisted_max_norm_deviation": float(np.max(np.abs(assisted["norm_history"] - 1.0))),
                "assisted_raw_max_norm_deviation": float(
                    np.max(np.abs(assisted.get("raw_norm_history", assisted["norm_history"]) - 1.0))
                ),
                "max_delta_schedule": float(np.max(assisted["delta_grid"])),
                "delta_schedule_area": float(np.trapezoid(assisted["delta_grid"], dx=1.0 / int(args.n_steps))),
                "max_auxiliary_envelope": float(np.max(assisted["auxiliary_envelope_grid"])),
                "auxiliary_envelope_area": float(
                    np.trapezoid(assisted["auxiliary_envelope_grid"], dx=1.0 / int(args.n_steps))
                ),
                "evolution_method": "continuous_ode_qaoa_mixer_assisted_baseline_schedule",
                "problem_hamiltonian": "baseline_penalized_maximum_clique_qubo",
                "standard_driver": "transverse_field_driver",
                "auxiliary_mixer": "qaoa_style_graph_aware_controlled_flip_mixer",
                "initial_state": "plus_state",
                "ode_method": args.ode_method,
                "ode_rtol": float(args.ode_rtol),
                "ode_atol": float(args.ode_atol),
            }
        )
        row.update(mixer_stats)

        add_metrics_with_prefix(row, "base", base_metrics)
        add_metrics_with_prefix(row, "assisted", assisted_metrics)

        row["delta_energy"] = row["assisted_final_expected_energy"] - row["base_final_expected_energy"]
        row["delta_energy_variance"] = row["assisted_energy_variance"] - row["base_energy_variance"]
        row["delta_expected_feasible_energy"] = (
            row["assisted_expected_feasible_energy"] - row["base_expected_feasible_energy"]
        )
        row["delta_feasible_energy_variance"] = (
            row["assisted_feasible_energy_variance"] - row["base_feasible_energy_variance"]
        )
        row["delta_p_ground"] = (
            row["assisted_ground_state_probability"] - row["base_ground_state_probability"]
        )
        row["delta_p_feas"] = row["assisted_feasibility_probability"] - row["base_feasibility_probability"]

        result_rows.append(row)
        append_energy_history(histories, assisted, instance_base, "qaoa_mixer_assisted", rho)
        schedules.extend(schedule_rows(instance_base, args.n_steps, "qaoa_mixer_assisted", rho))

        if args.save_probabilities:
            append_probability_rows(
                probability_rows,
                assisted["final_state"],
                bitstrings,
                cost_energies,
                instance_base,
                "qaoa_mixer_assisted",
                rho,
                args.probability_tol,
            )

    return result_rows, histories, schedules, mixer_rows, probability_rows


# write all output tables.
def save_outputs(
    results: pd.DataFrame,
    histories: pd.DataFrame,
    schedules: pd.DataFrame,
    mixer_specification: pd.DataFrame,
    probabilities: pd.DataFrame,
    args,
) -> None:
    out_dir = FORMULATION_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "formulation4_results.csv", index=False)
    histories.to_csv(out_dir / "energy_histories.csv", index=False)
    schedules.to_csv(out_dir / "schedules.csv", index=False)
    mixer_specification.to_csv(out_dir / "auxiliary_mixer_specification.csv", index=False)

    if not probabilities.empty:
        probabilities.to_csv(out_dir / "probabilities.csv", index=False)

    metrics = [
        "delta_energy",
        "delta_energy_variance",
        "delta_p_ground",
        "delta_expected_feasible_energy",
        "delta_feasible_energy_variance",
        "delta_p_feas",
    ]

    rho_summary_table(results).to_csv(out_dir / "summary_by_rho.csv", index=False)
    overall_summary_table(results).to_csv(out_dir / "overall_summary.csv", index=False)
    summarize_by_group(results, ["n"]).to_csv(out_dir / "summary_by_n.csv", index=False)
    summarize_by_group(results, ["p"]).to_csv(out_dir / "summary_by_p.csv", index=False)
    summarize_by_group(results, ["rho", "n"]).to_csv(
        out_dir / "summary_by_rho_and_n.csv",
        index=False,
    )
    summarize_by_group(results, ["rho", "p"]).to_csv(
        out_dir / "summary_by_rho_and_p.csv",
        index=False,
    )
    confidence_table(results, metrics, n_boot=args.n_boot).to_csv(
        out_dir / "confidence_intervals.csv",
        index=False,
    )
    best_by_instance_table(results).to_csv(out_dir / "best_by_instance.csv", index=False)

    args_dict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    metadata = {
        "created_at_unix": time.time(),
        "repository_dir": str(REPOSITORY_DIR),
        "formulation_dir": str(FORMULATION_DIR),
        "instances_dir": str(args.instances_dir),
        "args": args_dict,
        "package_versions": package_versions(),
    }

    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# parse command-line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Formulation 4 experiments.")
    parser.add_argument("--instances-dir", type=Path, default=REPOSITORY_DIR / "ER_instances")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-values", type=int, nargs="*", default=None)
    parser.add_argument("--p-values", type=float, nargs="*", default=None)
    parser.add_argument("--rho-values", type=float, nargs="*", default=[0.25, 0.50, 1.00, 2.00])
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--ode-rtol", type=float, default=1e-8)
    parser.add_argument("--ode-atol", type=float, default=1e-10)
    parser.add_argument("--ode-method", type=str, default="DOP853")
    parser.add_argument("--eig-tol", type=float, default=1e-8)
    parser.add_argument("--eig-maxiter", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--probability-tol", type=float, default=1e-12)
    return parser.parse_args()


# run all selected instances.
def main() -> None:
    args = parse_args()
    paths = collect_instance_paths(args.instances_dir, args.n_values, args.p_values)

    if not paths:
        raise FileNotFoundError(f"No edgelist files found in {args.instances_dir}.")

    all_results = []
    all_histories = []
    all_schedules = []
    all_mixer_specs = []
    all_probabilities = []

    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path.name}", flush=True)
        results, histories, schedules, mixer_specs, probabilities = run_single_instance(path, args)
        all_results.extend(results)
        all_histories.extend(histories)
        all_schedules.extend(schedules)
        all_mixer_specs.extend(mixer_specs)
        all_probabilities.extend(probabilities)

    save_outputs(
        results=pd.DataFrame(all_results),
        histories=pd.DataFrame(all_histories),
        schedules=pd.DataFrame(all_schedules),
        mixer_specification=pd.DataFrame(all_mixer_specs),
        probabilities=pd.DataFrame(all_probabilities),
        args=args,
    )

    print(f"Saved outputs to {FORMULATION_DIR / args.output_dir}")


if __name__ == "__main__":
    main()
