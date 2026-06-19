from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# create the uniform superposition state.
def plus_state(n: int) -> np.ndarray:
    state = np.ones(2**int(n), dtype=complex)
    return state / np.linalg.norm(state)


# compute the expected QUBO cost of a state.
def expected_cost(state: np.ndarray, cost_energies: np.ndarray) -> float:
    probabilities = np.abs(state) ** 2
    return float(np.real(np.dot(probabilities, cost_energies)))


# create endpoint values for the linear annealing schedule.
def coefficient_schedule_grid(N_steps: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0, int(N_steps) + 1, dtype=float) / float(N_steps)
    alpha = grid
    beta = 1.0 - grid
    return alpha, beta


# return one schedule value per completed step.
def coefficient_schedules(N_steps: int) -> tuple[np.ndarray, np.ndarray]:
    alpha_grid, beta_grid = coefficient_schedule_grid(N_steps)
    return alpha_grid[1:].copy(), beta_grid[1:].copy()


# precompute the index obtained by flipping each qubit.
def make_flip_indices(n: int) -> list[np.ndarray]:
    dim = 2**int(n)
    indices = np.arange(dim)
    return [indices ^ (1 << (int(n) - 1 - qubit)) for qubit in range(int(n))]


# compute H_M psi for H_M = -sum_i X_i.
def mixer_hamiltonian_times_state(
    state: np.ndarray,
    flip_indices: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(state, dtype=complex)

    for flipped in flip_indices:
        out -= state[flipped]

    return out


# compute H(t) psi = alpha H_P psi + beta H_M psi.
def annealing_hamiltonian_times_state(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    alpha: float,
    beta: float,
) -> np.ndarray:
    problem_part = float(alpha) * cost_energies * state
    mixer_part = float(beta) * mixer_hamiltonian_times_state(state, flip_indices)
    return problem_part + mixer_part


# linearly interpolate one coefficient inside an interval.
def interpolate_endpoint_values(t: float, dt: float, left: float, right: float) -> float:
    if float(dt) == 0.0:
        return float(right)

    fraction = float(t) / float(dt)
    return float(left) + fraction * (float(right) - float(left))


# evolve one interval with linearly interpolated coefficients.
def evolve_under_interval_hamiltonian(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
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

        hamiltonian_state = annealing_hamiltonian_times_state(
            y,
            cost_energies,
            flip_indices,
            alpha_t,
            beta_t,
        )

        return -1j * hamiltonian_state

    # integrate only up to the interval endpoint.
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
        raise RuntimeError(f"Continuous integration failed: {sol.message}")

    out = np.asarray(sol.y[:, -1], dtype=complex)
    raw_norm = float(np.linalg.norm(out))

    # renormalize to correct small numerical drift.
    if normalize and raw_norm > 0:
        out = out / raw_norm

    return out, raw_norm


# simulate the baseline continuous-time annealing process.
def simulate_baseline(
    cost_energies: np.ndarray,
    n: int,
    T: float,
    N_steps: int,
    angle_scale: float = 1.0,
    initial_state: np.ndarray | None = None,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
) -> dict:
    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    alpha_grid, beta_grid = coefficient_schedule_grid(N_steps)

    state = plus_state(n) if initial_state is None else np.asarray(initial_state, dtype=complex).copy()
    flip_indices = make_flip_indices(n)

    energy_history = [expected_cost(state, cost_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]

    for step_index in range(1, int(N_steps) + 1):
        state, raw_norm = evolve_under_interval_hamiltonian(
            state=state,
            cost_energies=cost_energies,
            flip_indices=flip_indices,
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
        "angle_scale": float(angle_scale),
        "evolution_method": "continuous_ode_linear_schedule",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
    }