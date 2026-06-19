from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from base_setup.statevector import (
    coefficient_schedule_grid,
    expected_cost,
    interpolate_endpoint_values,
    make_flip_indices,
)


# return the complement edges used in the max clique penalty.
def complement_edges(n: int, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    edge_set = {tuple(sorted(edge)) for edge in edges}

    return {
        (i, j)
        for i in range(int(n))
        for j in range(i + 1, int(n))
        if (i, j) not in edge_set
    }


# solve the SDP relaxation used to obtain the warm-start vector.
def solve_sdp_warm_start(
    n: int,
    edges: set[tuple[int, int]],
    penalty: float = 2.0,
    solver: str = "SCS",
    eps_solver: float = 1e-6,
    max_iters: int = 20000,
    verbose: bool = False,
) -> dict[str, Any]:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("Install cvxpy and scs before running Formulation 1.") from exc

    non_edges = complement_edges(n, edges)

    x = cp.Variable(int(n))
    X = cp.Variable((int(n), int(n)), symmetric=True)

    x_col = cp.reshape(x, (int(n), 1), order="C")
    x_row = cp.reshape(x, (1, int(n)), order="C")
    lifted = cp.bmat([[np.ones((1, 1)), x_row], [x_col, X]])

    constraints = [lifted >> 0, x >= -eps_solver, x <= 1.0 + eps_solver]

    for i in range(int(n)):
        constraints.append(X[i, i] == x[i])

    constraints += [
        X >= -eps_solver,
        X <= x_col + eps_solver,
        X <= x_row + eps_solver,
        X >= x_col + x_row - 1.0 - eps_solver,
    ]

    if non_edges:
        penalty_expr = cp.sum([X[i, j] for i, j in non_edges])
    else:
        penalty_expr = 0.0

    objective = cp.Minimize(-cp.sum(x) + float(penalty) * penalty_expr)
    problem = cp.Problem(objective, constraints)

    solve_kwargs = {"verbose": bool(verbose)}

    if solver.upper() == "SCS":
        solve_kwargs.update({"eps": float(eps_solver), "max_iters": int(max_iters)})

    problem.solve(solver=solver, **solve_kwargs)

    return {
        "x_value": None if x.value is None else np.asarray(x.value, dtype=float).reshape(-1),
        "objective_value": None if problem.value is None else float(problem.value),
        "status": problem.status,
        "solver": solver,
        "solve_time": getattr(problem.solver_stats, "solve_time", None),
        "num_iters": getattr(problem.solver_stats, "num_iters", None),
    }


# clip the SDP vector away from zero and one.
def clip_warm_start_vector(x_value: np.ndarray, eps_clip: float = 1e-6) -> np.ndarray:
    if x_value is None:
        raise ValueError("Cannot build a warm-start vector from a missing SDP solution.")

    return np.clip(
        np.asarray(x_value, dtype=float).reshape(-1),
        float(eps_clip),
        1.0 - float(eps_clip),
    )


# convert the warm-start vector to rotation angles.
def warm_start_angles(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return 2.0 * np.arcsin(np.sqrt(c))


# create the product warm-start initial state.
def warm_start_state(c: np.ndarray) -> np.ndarray:
    state = np.asarray([1.0 + 0.0j], dtype=complex)

    for ci in np.asarray(c, dtype=float):
        qubit_state = np.asarray(
            [np.sqrt(1.0 - ci), np.sqrt(ci)],
            dtype=complex,
        )
        state = np.kron(state, qubit_state)

    return state / np.linalg.norm(state)


# precompute the Z eigenvalue of each basis state for each qubit.
def make_z_signs(n: int) -> list[np.ndarray]:
    dim = 2**int(n)
    indices = np.arange(dim)
    signs = []

    for qubit in range(int(n)):
        bits = (indices >> (int(n) - 1 - qubit)) & 1
        signs.append(1.0 - 2.0 * bits.astype(float))

    return signs


# compute H_M^WS psi for the warm-start mixer.
def warm_start_mixer_hamiltonian_times_state(
    state: np.ndarray,
    flip_indices: list[np.ndarray],
    z_signs: list[np.ndarray],
    theta: np.ndarray,
) -> np.ndarray:
    out = np.zeros_like(state, dtype=complex)

    for qubit, theta_i in enumerate(np.asarray(theta, dtype=float)):
        sin_theta = float(np.sin(theta_i))
        cos_theta = float(np.cos(theta_i))

        out -= sin_theta * state[flip_indices[qubit]]
        out -= cos_theta * z_signs[qubit] * state

    return out


# compute H^WS(t) psi = alpha H_P psi + beta H_M^WS psi.
def warm_start_hamiltonian_times_state(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    z_signs: list[np.ndarray],
    theta: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    problem_part = float(alpha) * cost_energies * state
    mixer_part = float(beta) * warm_start_mixer_hamiltonian_times_state(
        state,
        flip_indices,
        z_signs,
        theta,
    )

    return problem_part + mixer_part


# evolve one warm-start interval with linearly interpolated coefficients.
def evolve_under_warm_start_interval_hamiltonian(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    z_signs: list[np.ndarray],
    theta: np.ndarray,
    alpha_left: float,
    alpha_right: float,
    beta_left: float,
    beta_right: float,
    dt: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "DOP853",
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    y0 = np.asarray(state, dtype=complex)
    interval_length = float(dt)

    if interval_length == 0.0:
        return y0.copy(), float(np.linalg.norm(y0))

    # right-hand side of the Schrodinger equation.
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        alpha_t = interpolate_endpoint_values(t, interval_length, alpha_left, alpha_right)
        beta_t = interpolate_endpoint_values(t, interval_length, beta_left, beta_right)

        hamiltonian_state = warm_start_hamiltonian_times_state(
            y,
            cost_energies,
            flip_indices,
            z_signs,
            theta,
            alpha_t,
            beta_t,
        )

        return -1j * hamiltonian_state

    # request only the endpoint state from the adaptive solver.
    sol = solve_ivp(
        rhs,
        (0.0, interval_length),
        y0,
        method=method,
        rtol=float(rtol),
        atol=float(atol),
        t_eval=[interval_length],
    )

    if not sol.success:
        raise RuntimeError(f"Warm-start integration failed: {sol.message}")

    out = np.asarray(sol.y[:, -1], dtype=complex)
    raw_norm = float(np.linalg.norm(out))

    # renormalize to correct small numerical drift.
    if normalize and raw_norm > 0:
        out = out / raw_norm

    return out, raw_norm


# simulate warm-start annealing with the baseline endpoint schedule.
def simulate_warm_start_annealing(
    cost_energies: np.ndarray,
    c: np.ndarray,
    T: float,
    N_steps: int,
    angle_scale: float = 1.0,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
) -> dict:
    c = np.asarray(c, dtype=float)
    n = int(c.size)

    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    alpha_grid, beta_grid = coefficient_schedule_grid(N_steps)

    state = warm_start_state(c)
    theta = warm_start_angles(c)
    flip_indices = make_flip_indices(n)
    z_signs = make_z_signs(n)

    energy_history = [expected_cost(state, cost_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]

    for step_index in range(1, int(N_steps) + 1):
        state, raw_norm = evolve_under_warm_start_interval_hamiltonian(
            state=state,
            cost_energies=cost_energies,
            flip_indices=flip_indices,
            z_signs=z_signs,
            theta=theta,
            alpha_left=float(alpha_grid[step_index - 1]),
            alpha_right=float(alpha_grid[step_index]),
            beta_left=float(beta_grid[step_index - 1]),
            beta_right=float(beta_grid[step_index]),
            dt=effective_dt,
            rtol=ode_rtol,
            atol=ode_atol,
            method=ode_method,
            normalize=True,
        )

        energy_history.append(expected_cost(state, cost_energies))
        norm_history.append(float(np.linalg.norm(state)))
        raw_norm_history.append(raw_norm)

    return {
        "final_state": state,
        "energy_history": np.asarray(energy_history),
        "norm_history": np.asarray(norm_history),
        "raw_norm_history": np.asarray(raw_norm_history),
        "dt": dt,
        "effective_dt": effective_dt,
        "alpha_schedule": alpha_grid[1:].copy(),
        "beta_schedule": beta_grid[1:].copy(),
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "c": c,
        "theta": theta,
        "angle_scale": float(angle_scale),
        "evolution_method": "continuous_ode_linear_schedule",
        "mixer_type": "warm_start_mixer",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
    }
