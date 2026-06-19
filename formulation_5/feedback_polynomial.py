from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from base_setup.statevector import (
    coefficient_schedule_grid,
    evolve_under_interval_hamiltonian,
    expected_cost,
    make_flip_indices,
    mixer_hamiltonian_times_state,
    plus_state,
)


@dataclass(frozen=True)
class PolynomialMinimizationResult:
    beta: float
    value: float
    selected_kind: str
    derivative_abs_at_beta: float
    candidate_count: int


# compute the FALQON commutator signal.
def falqon_commutator_signal(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
) -> float:
    commutator_state = np.zeros_like(state, dtype=complex)

    for flipped in flip_indices:
        commutator_state += (
            1j * (cost_energies - cost_energies[flipped]) * state[flipped]
        )

    value = np.vdot(state, commutator_state)
    return float(np.real_if_close(value, tol=1000).real)


# remove negligible highest-order coefficients.
def _trim_polynomial(coefficients: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).copy()

    while coeffs.size > 1 and abs(coeffs[-1]) < tol:
        coeffs = coeffs[:-1]

    return coeffs


# evaluate a polynomial written in increasing powers of beta.
def evaluate_polynomial(coefficients: np.ndarray, beta: float) -> float:
    return float(np.polynomial.polynomial.polyval(float(beta), coefficients))


# compute derivative coefficients.
def derivative_coefficients(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)

    if coefficients.size <= 1:
        return np.asarray([0.0], dtype=float)

    return np.asarray(
        [power * coefficients[power] for power in range(1, coefficients.size)],
        dtype=float,
    )


# find real stationary points inside an interval.
def real_stationary_points(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
    tol: float = 1e-9,
) -> list[float]:
    deriv = _trim_polynomial(derivative_coefficients(coefficients), tol=tol)

    if deriv.size <= 1:
        return []

    roots = np.roots(deriv[::-1])
    points: list[float] = []

    for root in roots:
        if abs(root.imag) <= tol * max(1.0, abs(root.real)):
            value = float(root.real)

            if lower - tol <= value <= upper + tol:
                points.append(float(np.clip(value, lower, upper)))

    points = sorted(set(round(point, 14) for point in points))
    return [float(point) for point in points]


# minimize a polynomial over a closed interval.
def minimize_polynomial_on_interval(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
) -> PolynomialMinimizationResult:
    lower = float(lower)
    upper = float(upper)

    if lower > upper:
        raise ValueError("lower must be <= upper.")

    coeffs = _trim_polynomial(coefficients)

    candidates = [lower, upper]
    candidates.extend(real_stationary_points(coeffs, lower, upper))
    candidates = sorted(set(round(float(candidate), 14) for candidate in candidates))

    values = [evaluate_polynomial(coeffs, candidate) for candidate in candidates]

    best_index = int(np.argmin(values))
    beta = float(candidates[best_index])
    value = float(values[best_index])

    if abs(beta - lower) <= 1e-10:
        selected_kind = "lower_endpoint"
    elif abs(beta - upper) <= 1e-10:
        selected_kind = "upper_endpoint"
    else:
        selected_kind = "stationary_point"

    deriv = derivative_coefficients(coeffs)

    return PolynomialMinimizationResult(
        beta=beta,
        value=value,
        selected_kind=selected_kind,
        derivative_abs_at_beta=abs(evaluate_polynomial(deriv, beta)),
        candidate_count=len(candidates),
    )


# compute (alpha H_P + beta H_M) times a beta-polynomial state.
def annealing_hamiltonian_times_vector_polynomial(
    vector_poly: list[np.ndarray],
    alpha: float,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
) -> list[np.ndarray]:
    out = [
        np.zeros_like(vector_poly[0], dtype=complex)
        for _ in range(len(vector_poly) + 1)
    ]

    for degree, vector in enumerate(vector_poly):
        out[degree] += float(alpha) * cost_energies * vector
        out[degree + 1] += mixer_hamiltonian_times_state(vector, flip_indices)

    return out


# build the local Taylor polynomial in beta.
def local_energy_taylor_polynomial(
    state: np.ndarray,
    cost_energies: np.ndarray,
    flip_indices: list[np.ndarray],
    alpha: float,
    dt: float,
    degree: int,
) -> np.ndarray:
    q = int(degree)

    if q < 0:
        raise ValueError("degree must be nonnegative.")

    powers: list[list[np.ndarray]] = [[np.asarray(state, dtype=complex)]]

    for _ in range(1, q + 1):
        powers.append(
            annealing_hamiltonian_times_vector_polynomial(
                vector_poly=powers[-1],
                alpha=alpha,
                cost_energies=cost_energies,
                flip_indices=flip_indices,
            )
        )

    coeffs = np.zeros(q + 1, dtype=complex)

    for left_power in range(q + 1):
        for right_power in range(q + 1 - left_power):
            prefactor = (
                ((1j * dt) ** left_power / math.factorial(left_power))
                * ((-1j * dt) ** right_power / math.factorial(right_power))
            )

            for left_degree, left_vector in enumerate(powers[left_power]):
                for right_degree, right_vector in enumerate(powers[right_power]):
                    beta_degree = left_degree + right_degree

                    if beta_degree <= q:
                        coeffs[beta_degree] += prefactor * np.vdot(
                            left_vector,
                            cost_energies * right_vector,
                        )

    return np.real_if_close(coeffs, tol=1000).real.astype(float)


# construct the admissible beta interval.
def admissible_interval(
    previous_beta: float,
    envelope_beta: float,
    restriction_type: str,
    restriction_value: float | None,
    eps_beta: float,
) -> tuple[float, float, bool]:
    lower = float(eps_beta)
    upper = float(envelope_beta)
    repaired = False

    if restriction_type == "none":
        pass
    elif restriction_type == "absolute":
        rho = float(restriction_value)
        lower = max(lower, float(previous_beta) - rho)
        upper = min(upper, float(previous_beta) + rho)
    elif restriction_type == "relative":
        tau = float(restriction_value)
        lower = max(lower, (1.0 - tau) * float(previous_beta))
        upper = min(upper, (1.0 + tau) * float(previous_beta))
    else:
        raise ValueError(f"Unknown restriction_type: {restriction_type}")

    if lower > upper:
        lower = upper
        repaired = True

    return float(lower), float(upper), repaired


# simulate polynomial feedback annealing.
def simulate_polynomial_feedback(
    cost_energies: np.ndarray,
    n: int,
    T: float,
    N_steps: int,
    degree: int,
    restriction_type: str,
    restriction_value: float | None,
    eps_beta: float = 1e-8,
    beta0: float = 1.0,
    angle_scale: float = 1.0,
    initial_state: np.ndarray | None = None,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
) -> dict:
    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    state = (
        plus_state(n)
        if initial_state is None
        else np.asarray(initial_state, dtype=complex).copy()
    )

    flip_indices = make_flip_indices(n)

    alpha_grid, envelope_grid = coefficient_schedule_grid(N_steps)
    alpha_schedule = alpha_grid[1:].copy()
    envelope_schedule = envelope_grid[1:].copy()

    applied_beta_schedule = np.zeros(int(N_steps), dtype=float)
    full_beta_schedule = np.zeros(int(N_steps) + 1, dtype=float)
    full_beta_schedule[0] = float(beta0)

    energy_history = [expected_cost(state, cost_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]
    schedule_rows = []

    previous_beta = float(beta0)

    for step_index in range(1, int(N_steps) + 1):
        alpha_left = float(alpha_grid[step_index - 1])
        alpha_k = float(alpha_grid[step_index])

        beta_left = float(previous_beta)

        envelope_beta_left = float(envelope_grid[step_index - 1])
        envelope_beta = float(envelope_grid[step_index])

        if step_index == int(N_steps):
            beta_k = 0.0
            lower = 0.0
            upper = 0.0
            repaired = False
            selected_kind = "forced_final_zero"
            poly_value = float("nan")
            poly_at_envelope = float("nan")
            derivative_abs = float("nan")
            candidate_count = 1
            first_order_signal = falqon_commutator_signal(
                state,
                cost_energies,
                flip_indices,
            )
            coeffs = np.full(int(degree) + 1, np.nan)

        else:
            lower, upper, repaired = admissible_interval(
                previous_beta=previous_beta,
                envelope_beta=envelope_beta,
                restriction_type=restriction_type,
                restriction_value=restriction_value,
                eps_beta=eps_beta,
            )

            coeffs = local_energy_taylor_polynomial(
                state=state,
                cost_energies=cost_energies,
                flip_indices=flip_indices,
                alpha=alpha_k,
                dt=effective_dt,
                degree=degree,
            )

            best = minimize_polynomial_on_interval(coeffs, lower, upper)

            beta_k = best.beta
            selected_kind = best.selected_kind
            poly_value = best.value
            derivative_abs = best.derivative_abs_at_beta
            candidate_count = best.candidate_count

            beta_reference = float(np.clip(envelope_beta, lower, upper))
            poly_at_envelope = evaluate_polynomial(coeffs, beta_reference)

            first_order_signal = falqon_commutator_signal(
                state,
                cost_energies,
                flip_indices,
            )

        state, raw_norm = evolve_under_interval_hamiltonian(
            state=state,
            cost_energies=cost_energies,
            flip_indices=flip_indices,
            alpha_left=alpha_left,
            alpha_right=alpha_k,
            beta_left=beta_left,
            beta_right=beta_k,
            dt=effective_dt,
            rtol=ode_rtol,
            atol=ode_atol,
            method=ode_method,
            normalize=True,
        )

        energy_after = expected_cost(state, cost_energies)

        applied_beta_schedule[step_index - 1] = beta_k
        full_beta_schedule[step_index] = beta_k

        schedule_rows.append(
            {
                "step": step_index,
                "alpha": alpha_k,
                "alpha_left": alpha_left,
                "alpha_right": alpha_k,
                "beta": float(beta_k),
                "beta_left": beta_left,
                "beta_right": float(beta_k),
                "beta_envelope": envelope_beta,
                "beta_envelope_left": envelope_beta_left,
                "interval_lower": float(lower),
                "interval_upper": float(upper),
                "interval_repaired": bool(repaired),
                "selected_kind": selected_kind,
                "local_poly_value": float(poly_value),
                "local_poly_at_envelope_or_clipped": float(poly_at_envelope),
                "predicted_gain_vs_envelope": float(poly_value - poly_at_envelope)
                if np.isfinite(poly_value) and np.isfinite(poly_at_envelope)
                else float("nan"),
                "derivative_abs_at_beta": float(derivative_abs),
                "candidate_count": int(candidate_count),
                "first_order_signal_A": float(first_order_signal),
                "energy_after_step": float(energy_after),
                "raw_norm_after_ode": float(raw_norm),
                **{f"poly_coeff_{j}": float(coeffs[j]) for j in range(len(coeffs))},
            }
        )

        energy_history.append(energy_after)
        norm_history.append(float(np.linalg.norm(state)))
        raw_norm_history.append(raw_norm)

        previous_beta = float(beta_k)

    return {
        "final_state": state,
        "energy_history": np.asarray(energy_history),
        "norm_history": np.asarray(norm_history),
        "raw_norm_history": np.asarray(raw_norm_history),
        "dt": dt,
        "effective_dt": effective_dt,
        "alpha_schedule": alpha_schedule,
        "beta_envelope_schedule": envelope_schedule,
        "applied_beta_schedule": applied_beta_schedule,
        "full_beta_schedule": full_beta_schedule,
        "schedule_rows": schedule_rows,
        "degree": int(degree),
        "restriction_type": restriction_type,
        "restriction_value": restriction_value,
        "angle_scale": float(angle_scale),
        "evolution_method": "continuous_ode_linear_schedule",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
    }
