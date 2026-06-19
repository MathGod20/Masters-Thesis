from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORMULATION_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = FORMULATION_DIR.parent

if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from analysis import best_by_instance_table, confidence_table, method_ranking_table, summarize_by_group
from feedback_polynomial import simulate_polynomial_feedback
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


# create a readable value label for filenames and tables.
def value_label(restriction_type: str, value: float | None) -> str:
    if restriction_type == "none" or value is None:
        return "none"

    return f"{float(value):.6g}".replace(".", "p")


# create a unique method identifier.
def method_id(degree: int, restriction_type: str, value: float | None) -> str:
    if restriction_type == "none":
        return f"q{degree}_unrestricted"

    short = "abs" if restriction_type == "absolute" else "rel"
    return f"q{degree}_{short}_{value_label(restriction_type, value)}"


# create a readable restriction label.
def restriction_label(restriction_type: str, value: float | None) -> str:
    if restriction_type == "none":
        return "envelope only"

    symbol = "rho" if restriction_type == "absolute" else "tau"
    return f"{restriction_type} {symbol}={float(value):.6g}"


# build all feedback method configurations.
def build_method_configs(args) -> list[dict[str, Any]]:
    configs = []

    if args.include_unrestricted:
        for degree in args.degrees:
            configs.append(
                {
                    "degree": degree,
                    "restriction_type": "none",
                    "restriction_value": None,
                }
            )

    for degree in args.degrees:
        for rho in args.absolute_rhos:
            configs.append(
                {
                    "degree": degree,
                    "restriction_type": "absolute",
                    "restriction_value": float(rho),
                }
            )

        for tau in args.relative_taus:
            configs.append(
                {
                    "degree": degree,
                    "restriction_type": "relative",
                    "restriction_value": float(tau),
                }
            )

    if args.method_limit is not None:
        configs = configs[: args.method_limit]

    return configs


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


# summarize the selected beta schedule.
def beta_diagnostics(
    schedule_rows: list[dict],
    full_beta_schedule: np.ndarray,
    dt: float,
) -> dict:
    schedule = np.asarray(full_beta_schedule, dtype=float)
    diffs = np.diff(schedule)

    intermediate = [row for row in schedule_rows if row["step"] < len(schedule) - 1]
    kinds = [row["selected_kind"] for row in intermediate]

    predicted = np.asarray(
        [row["predicted_gain_vs_envelope"] for row in intermediate],
        dtype=float,
    )
    signals = np.asarray(
        [row["first_order_signal_A"] for row in intermediate],
        dtype=float,
    )
    derivative_abs = np.asarray(
        [row["derivative_abs_at_beta"] for row in intermediate],
        dtype=float,
    )

    return {
        "beta_mean": float(np.mean(schedule[1:])),
        "beta_min": float(np.min(schedule[1:])),
        "beta_max": float(np.max(schedule[1:])),
        "beta_std": float(np.std(schedule[1:], ddof=1)) if len(schedule) > 2 else 0.0,
        "beta_area": float(np.sum(schedule[1:]) * dt),
        "beta_total_variation": float(np.sum(np.abs(diffs))),
        "beta_max_abs_step_change": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
        "beta_increase_count": int(np.sum(diffs > 1e-10)),
        "beta_decrease_count": int(np.sum(diffs < -1e-10)),
        "beta_last_before_zero": float(schedule[-2]) if len(schedule) >= 2 else float("nan"),
        "stationary_selected_rate": float(np.mean([kind == "stationary_point" for kind in kinds])) if kinds else float("nan"),
        "lower_endpoint_selected_rate": float(np.mean([kind == "lower_endpoint" for kind in kinds])) if kinds else float("nan"),
        "upper_endpoint_selected_rate": float(np.mean([kind == "upper_endpoint" for kind in kinds])) if kinds else float("nan"),
        "endpoint_selected_rate": float(np.mean([kind in {"lower_endpoint", "upper_endpoint"} for kind in kinds])) if kinds else float("nan"),
        "mean_predicted_gain_vs_envelope": float(np.nanmean(predicted)) if predicted.size else float("nan"),
        "mean_abs_first_order_signal_A": float(np.mean(np.abs(signals))) if signals.size else float("nan"),
        "mean_derivative_abs_at_beta": float(np.nanmean(derivative_abs)) if derivative_abs.size else float("nan"),
        "interval_repaired_count": int(sum(bool(row["interval_repaired"]) for row in intermediate)),
    }


# add metrics to a row using a prefix.
def add_metrics_with_prefix(row: dict, prefix: str, metrics: dict) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


# run baseline and all feedback variants on one instance.
def run_single_instance(
    path: Path,
    method_configs: list[dict[str, Any]],
    args,
) -> tuple[list[dict], list[dict], list[dict]]:
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
        "num_edges": len(edges),
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
        "ode_method": args.ode_method,
        "ode_rtol": float(args.ode_rtol),
        "ode_atol": float(args.ode_atol),
    }
    add_metrics_with_prefix(instance_base, "base", base_metrics)

    histories = []
    for step, energy in enumerate(base["energy_history"]):
        histories.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": "baseline",
                "step": step,
                "expected_energy": float(energy),
            }
        )

    schedules = []
    alpha_grid, beta_grid = coefficient_schedule_grid(args.n_steps)

    schedules.append(
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
            "beta_envelope": float(beta_grid[0]),
            "beta_envelope_left": np.nan,
        }
    )

    for step in range(1, args.n_steps + 1):
        schedules.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": "baseline",
                "step": step,
                "alpha": float(alpha_grid[step]),
                "alpha_left": float(alpha_grid[step - 1]),
                "alpha_right": float(alpha_grid[step]),
                "beta": float(beta_grid[step]),
                "beta_left": float(beta_grid[step - 1]),
                "beta_right": float(beta_grid[step]),
                "beta_envelope": float(beta_grid[step]),
                "beta_envelope_left": float(beta_grid[step - 1]),
            }
        )

    result_rows = []

    for config in method_configs:
        mid = method_id(
            config["degree"],
            config["restriction_type"],
            config["restriction_value"],
        )
        rlabel = restriction_label(config["restriction_type"], config["restriction_value"])
        vlabel = value_label(config["restriction_type"], config["restriction_value"])

        t0_feedback = time.perf_counter()
        feedback = simulate_polynomial_feedback(
            cost_energies=cost_energies,
            n=n,
            T=args.total_time,
            N_steps=args.n_steps,
            degree=config["degree"],
            restriction_type=config["restriction_type"],
            restriction_value=config["restriction_value"],
            eps_beta=args.eps_beta,
            beta0=args.beta0,
            angle_scale=args.angle_scale,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
            ode_method=args.ode_method,
        )
        feedback_time = time.perf_counter() - t0_feedback

        feedback_metrics = final_state_metrics(
            state=feedback["final_state"],
            cost_energies=cost_energies,
            bitstrings=bitstrings,
            edges=edges,
        )

        row = dict(instance_base)
        row.update(
            {
                "method_id": mid,
                "degree": int(config["degree"]),
                "restriction_type": config["restriction_type"],
                "restriction_value": config["restriction_value"]
                if config["restriction_value"] is not None
                else np.nan,
                "restriction_value_label": vlabel,
                "restriction_label": rlabel,
                "feedback_simulation_time": feedback_time,
                "feedback_max_norm_deviation": float(
                    np.max(np.abs(feedback["norm_history"] - 1.0))
                ),
                "feedback_raw_max_norm_deviation": float(
                    np.max(
                        np.abs(
                            feedback.get("raw_norm_history", feedback["norm_history"]) - 1.0
                        )
                    )
                ),
            }
        )

        add_metrics_with_prefix(row, "feedback", feedback_metrics)
        row.update(
            beta_diagnostics(
                feedback["schedule_rows"],
                feedback["full_beta_schedule"],
                feedback["dt"],
            )
        )

        row["delta_energy"] = row["feedback_final_expected_energy"] - row["base_final_expected_energy"]
        row["delta_energy_variance"] = row["feedback_energy_variance"] - row["base_energy_variance"]
        row["delta_expected_feasible_energy"] = (
            row["feedback_expected_feasible_energy"] - row["base_expected_feasible_energy"]
        )
        row["delta_feasible_energy_variance"] = (
            row["feedback_feasible_energy_variance"] - row["base_feasible_energy_variance"]
        )
        row["delta_p_ground"] = (
            row["feedback_ground_state_probability"] - row["base_ground_state_probability"]
        )
        row["delta_p_feas"] = (
            row["feedback_feasibility_probability"] - row["base_feasibility_probability"]
        )

        result_rows.append(row)

        for step, energy in enumerate(feedback["energy_history"]):
            histories.append(
                {
                    "instance_id": instance_base["instance_id"],
                    "n": instance_base["n"],
                    "p": instance_base["p"],
                    "replicate": instance_base["replicate"],
                    "method": mid,
                    "step": step,
                    "expected_energy": float(energy),
                }
            )

        schedules.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "method": mid,
                "step": 0,
                "alpha": 0.0,
                "beta": float(args.beta0),
                "beta_envelope": 1.0,
            }
        )

        for schedule_row in feedback["schedule_rows"]:
            schedule_item = dict(schedule_row)
            schedule_item.update(
                {
                    "instance_id": instance_base["instance_id"],
                    "n": instance_base["n"],
                    "p": instance_base["p"],
                    "replicate": instance_base["replicate"],
                    "method": mid,
                    "degree": int(config["degree"]),
                    "restriction_type": config["restriction_type"],
                    "restriction_value": config["restriction_value"]
                    if config["restriction_value"] is not None
                    else np.nan,
                    "restriction_value_label": vlabel,
                    "restriction_label": rlabel,
                }
            )
            schedules.append(schedule_item)

    return result_rows, histories, schedules


# parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline and polynomial feedback annealing for Maximum Clique."
    )

    parser.add_argument("--instances-dir", type=Path, default=REPOSITORY_DIR / "ER_instances")
    parser.add_argument("--results-dir", type=Path, default=FORMULATION_DIR / "results")
    parser.add_argument("--n-values", type=int, nargs="*", default=None)
    parser.add_argument("--p-values", type=float, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method-limit", type=int, default=None)
    parser.add_argument("--total-time", type=float, default=10.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--ode-method", type=str, default="DOP853")
    parser.add_argument("--ode-rtol", type=float, default=1e-8)
    parser.add_argument("--ode-atol", type=float, default=1e-10)
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--degrees", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--absolute-rhos", type=float, nargs="*", default=None)
    parser.add_argument("--relative-taus", type=float, nargs="*", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--include-unrestricted", action="store_true")
    parser.add_argument("--eps-beta", type=float, default=1e-8)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)

    return parser.parse_args()


# run the full experiment.
def main() -> None:
    args = parse_args()

    if args.absolute_rhos is None:
        args.absolute_rhos = [
            1.0 / args.n_steps,
            2.0 / args.n_steps,
            5.0 / args.n_steps,
            10.0 / args.n_steps,
        ]

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

    method_configs = build_method_configs(args)

    if not method_configs:
        raise ValueError("No feedback method configurations selected.")

    results = []
    histories = []
    schedules = []

    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}")
        rows, history_rows, schedule_rows = run_single_instance(path, method_configs, args)
        results.extend(rows)
        histories.extend(history_rows)
        schedules.extend(schedule_rows)

    results_df = pd.DataFrame(results)
    histories_df = pd.DataFrame(histories)
    schedules_df = pd.DataFrame(schedules)

    results_df.to_csv(tables_dir / "formulation5_per_instance_results.csv", index=False)
    histories_df.to_csv(raw_dir / "formulation5_energy_histories.csv", index=False)
    schedules_df.to_csv(raw_dir / "formulation5_beta_schedules.csv", index=False)

    summaries = [
        summarize_by_group(
            results_df.assign(overall="all"),
            ["overall", "method_id", "degree", "restriction_type", "restriction_value_label"],
        ),
        summarize_by_group(
            results_df,
            ["method_id", "degree", "restriction_type", "restriction_value_label", "n"],
        ),
        summarize_by_group(
            results_df,
            ["method_id", "degree", "restriction_type", "restriction_value_label", "p"],
        ),
        summarize_by_group(
            results_df,
            ["method_id", "degree", "restriction_type", "restriction_value_label", "n", "p"],
        ),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        tables_dir / "formulation5_grouped_summary.csv",
        index=False,
    )

    ranking_df = method_ranking_table(results_df)
    ranking_df.to_csv(tables_dir / "formulation5_method_ranking.csv", index=False)

    best_df = best_by_instance_table(results_df)
    best_df.to_csv(tables_dir / "formulation5_best_by_instance.csv", index=False)

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
    ).to_csv(tables_dir / "formulation5_bootstrap_ci.csv", index=False)

    metadata = vars(args).copy()
    metadata.update(
        {
            "num_instances_run": len(paths),
            "num_feedback_methods": len(method_configs),
            "method_configs": method_configs,
            "package_versions": package_versions(),
        }
    )

    with (logs_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    print(f"Saved tables to {tables_dir} and raw outputs to {raw_dir}.")


if __name__ == "__main__":
    main()
