from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from base_setup.statevector import (
    coefficient_schedule_grid,
    expected_cost,
    interpolate_endpoint_values,
    make_flip_indices,
    mixer_hamiltonian_times_state,
    plus_state,
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
        raise ImportError("Install cvxpy and scs before running Formulation 2.") from exc

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


# convert clipped warm-start values to signed guiding coefficients.
def signed_guiding_coefficients(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float).reshape(-1)
    return 2.0 * c - 1.0


# precompute Z_i eigenvalue signs for every computational basis state.
def make_z_signs(n: int) -> list[np.ndarray]:
    dim = 2**int(n)
    indices = np.arange(dim)
    signs = []

    for qubit in range(int(n)):
        bits = (indices >> (int(n) - 1 - qubit)) & 1
        signs.append(1.0 - 2.0 * bits.astype(float))

    return signs


# precompute the diagonal warm-guiding energy E_G(x) = sum_i q_i (1 - 2x_i).
def make_guiding_energies(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    n = int(q.size)
    dim = 2**n
    guiding_energies = np.zeros(dim, dtype=float)
    z_signs = make_z_signs(n)

    for qubit, qi in enumerate(q):
        guiding_energies += float(qi) * z_signs[qubit]

    return guiding_energies


# build the endpoint grid for eta_k = 4 rho s_k (1 - s_k).
def warm_guiding_schedule_grid(N_steps: int, rho: float) -> np.ndarray:
    s_grid = np.arange(0, int(N_steps) + 1, dtype=float) / float(N_steps)
    return 4.0 * float(rho) * s_grid * (1.0 - s_grid)


# compute H(t) psi = alpha H_P psi + beta H_D psi + eta H_G psi.
def warm_guiding_hamiltonian_times_state(
    state: np.ndarray,
    cost_energies: np.ndarray,
    guiding_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    alpha: float,
    beta: float,
    eta: float,
) -> np.ndarray:
    problem_part = float(alpha) * cost_energies * state
    driver_part = float(beta) * mixer_hamiltonian_times_state(state, flip_indices)
    guiding_part = float(eta) * guiding_energies * state

    return problem_part + driver_part + guiding_part


# evolve one warm-guided interval with linearly interpolated coefficients.
def evolve_under_warm_guiding_interval_hamiltonian(
    state: np.ndarray,
    cost_energies: np.ndarray,
    guiding_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    alpha_left: float,
    alpha_right: float,
    beta_left: float,
    beta_right: float,
    eta_left: float,
    eta_right: float,
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
        eta_t = interpolate_endpoint_values(t, interval_length, eta_left, eta_right)

        hamiltonian_state = warm_guiding_hamiltonian_times_state(
            y,
            cost_energies,
            guiding_energies,
            flip_indices,
            alpha_t,
            beta_t,
            eta_t,
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
        raise RuntimeError(f"Warm-guiding integration failed: {sol.message}")

    out = np.asarray(sol.y[:, -1], dtype=complex)
    raw_norm = float(np.linalg.norm(out))

    # renormalize to correct small numerical drift.
    if normalize and raw_norm > 0:
        out = out / raw_norm

    return out, raw_norm


# simulate Formulation 2 with the baseline alpha/beta schedules and a middle-peaked guiding field.
def simulate_warm_guiding_annealing(
    cost_energies: np.ndarray,
    q: np.ndarray,
    T: float,
    N_steps: int,
    rho: float,
    angle_scale: float = 1.0,
    initial_state: np.ndarray | None = None,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
) -> dict:
    q = np.asarray(q, dtype=float).reshape(-1)
    n = int(q.size)

    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    alpha_grid, beta_grid = coefficient_schedule_grid(N_steps)
    eta_grid = warm_guiding_schedule_grid(N_steps, rho)

    state = plus_state(n) if initial_state is None else np.asarray(initial_state, dtype=complex).copy()
    state = state / np.linalg.norm(state)

    flip_indices = make_flip_indices(n)
    guiding_energies = make_guiding_energies(q)

    energy_history = [expected_cost(state, cost_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]

    for step_index in range(1, int(N_steps) + 1):
        state, raw_norm = evolve_under_warm_guiding_interval_hamiltonian(
            state=state,
            cost_energies=cost_energies,
            guiding_energies=guiding_energies,
            flip_indices=flip_indices,
            alpha_left=float(alpha_grid[step_index - 1]),
            alpha_right=float(alpha_grid[step_index]),
            beta_left=float(beta_grid[step_index - 1]),
            beta_right=float(beta_grid[step_index]),
            eta_left=float(eta_grid[step_index - 1]),
            eta_right=float(eta_grid[step_index]),
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
        "eta_schedule": eta_grid[1:].copy(),
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "eta_grid": eta_grid,
        "q": q.copy(),
        "guiding_energies": guiding_energies,
        "rho": float(rho),
        "angle_scale": float(angle_scale),
        "evolution_method": "continuous_ode_linear_schedule",
        "guiding_type": "warm_guiding_diagonal_field",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
    }
