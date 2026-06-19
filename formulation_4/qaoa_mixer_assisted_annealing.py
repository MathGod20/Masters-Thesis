from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, eigsh

from base_setup.statevector import (
    expected_cost,
    interpolate_endpoint_values,
    make_flip_indices,
    mixer_hamiltonian_times_state,
    plus_state,
)


# create the bit masks for the non-neighbours of each vertex.
def non_neighbor_masks(n: int, edges: set[tuple[int, int]]) -> list[int]:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    masks: list[int] = []

    for i in range(int(n)):
        mask = 0

        for j in range(int(n)):
            if i == j:
                continue

            if tuple(sorted((i, j))) not in edge_set:
                mask |= 1 << (int(n) - 1 - j)

        masks.append(mask)

    return masks


# precompute where each allowed graph-aware flip sends amplitude.
def build_allowed_qaoa_mixer_flip_data(n: int, edges: set[tuple[int, int]]) -> list[dict]:
    dim = 2**int(n)
    indices = np.arange(dim, dtype=np.int64)
    masks = non_neighbor_masks(n, edges)
    flip_data: list[dict] = []

    for qubit, mask in enumerate(masks):
        flip_mask = 1 << (int(n) - 1 - qubit)
        allowed = (indices & int(mask)) == 0
        source_indices = indices[allowed]
        target_indices = source_indices ^ flip_mask

        flip_data.append(
            {
                "qubit": int(qubit),
                "non_neighbor_mask": int(mask),
                "source_indices": source_indices,
                "target_indices": target_indices,
                "allowed_count": int(source_indices.size),
            }
        )

    return flip_data


# compute H_M^QAOA psi without forming a dense matrix.
def qaoa_mixer_hamiltonian_times_state(
    state: np.ndarray,
    flip_data: list[dict],
) -> np.ndarray:
    out = np.zeros_like(state, dtype=complex)

    for data in flip_data:
        sources = data["source_indices"]
        targets = data["target_indices"]
        out[targets] -= state[sources]

    return out


# return the exact spectral norm of H_D = -sum_i X_i.
def standard_driver_spectral_norm(n: int) -> float:
    return float(int(n))


# estimate the spectral norm of the graph-aware QAOA-style mixer.
def qaoa_mixer_spectral_norm(
    n: int,
    flip_data: list[dict],
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
) -> float:
    dim = 2**int(n)

    if dim <= 1:
        return 0.0

    def matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=complex)
        return qaoa_mixer_hamiltonian_times_state(vector, flip_data)

    linear_map = LinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues = eigsh(
        linear_map,
        k=1,
        which="LM",
        return_eigenvectors=False,
        tol=float(eig_tol),
        maxiter=eig_maxiter,
    )

    return float(np.max(np.abs(eigenvalues)))


# create the problem and standard-driver endpoint schedules.
def formulation4_alpha_beta_schedule_grid(N_steps: int) -> tuple[np.ndarray, np.ndarray]:
    s_grid = np.arange(0, int(N_steps) + 1, dtype=float) / float(N_steps)

    alpha = s_grid.copy()
    beta = 1.0 - s_grid

    return alpha.astype(float), beta.astype(float)


# create the auxiliary envelope 4s(1-s).
def auxiliary_envelope_grid(N_steps: int) -> np.ndarray:
    s_grid = np.arange(0, int(N_steps) + 1, dtype=float) / float(N_steps)
    envelope = 4.0 * s_grid * (1.0 - s_grid)
    return envelope.astype(float)


# create endpoint values for the auxiliary schedule delta(s)=rho*4s(1-s).
def auxiliary_schedule_grid(N_steps: int, rho: float) -> np.ndarray:
    return float(rho) * auxiliary_envelope_grid(N_steps)


# compute H(t) psi = alpha H_P psi + beta H_D psi + delta H_M^QAOA psi.
def qaoa_mixer_assisted_hamiltonian_times_state(
    state: np.ndarray,
    cost_energies: np.ndarray,
    standard_flip_indices: list[np.ndarray],
    qaoa_flip_data: list[dict],
    alpha: float,
    beta: float,
    delta: float,
) -> np.ndarray:
    problem_part = float(alpha) * cost_energies * state
    standard_driver_part = float(beta) * mixer_hamiltonian_times_state(state, standard_flip_indices)
    qaoa_mixer_part = float(delta) * qaoa_mixer_hamiltonian_times_state(state, qaoa_flip_data)
    return problem_part + standard_driver_part + qaoa_mixer_part


# evolve one interval under the three-term Hamiltonian.
def evolve_under_qaoa_mixer_assisted_interval_hamiltonian(
    state: np.ndarray,
    cost_energies: np.ndarray,
    standard_flip_indices: list[np.ndarray],
    qaoa_flip_data: list[dict],
    alpha_left: float,
    alpha_right: float,
    beta_left: float,
    beta_right: float,
    delta_left: float,
    delta_right: float,
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
        delta_t = interpolate_endpoint_values(t, interval_length, delta_left, delta_right)
        hamiltonian_state = qaoa_mixer_assisted_hamiltonian_times_state(
            y,
            cost_energies,
            standard_flip_indices,
            qaoa_flip_data,
            alpha_t,
            beta_t,
            delta_t,
        )
        return -1j * hamiltonian_state

    # request only the interval endpoint from the adaptive solver.
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


# simulate the QAOA-mixer-assisted continuous-time annealing process.
def simulate_qaoa_mixer_assisted_annealing(
    n: int,
    edges: set[tuple[int, int]],
    cost_energies: np.ndarray,
    T: float,
    N_steps: int,
    rho: float,
    angle_scale: float = 1.0,
    initial_state: np.ndarray | None = None,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
    eig_tol: float = 1e-8,
    eig_maxiter: int | None = None,
    qaoa_flip_data: list[dict] | None = None,
    standard_driver_norm: float | None = None,
    qaoa_mixer_norm: float | None = None,
) -> dict:
    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    if qaoa_flip_data is None:
        qaoa_flip_data = build_allowed_qaoa_mixer_flip_data(n, edges)

    standard_norm = (
        standard_driver_spectral_norm(n)
        if standard_driver_norm is None
        else float(standard_driver_norm)
    )
    qaoa_norm = (
        qaoa_mixer_spectral_norm(
            n=n,
            flip_data=qaoa_flip_data,
            eig_tol=eig_tol,
            eig_maxiter=eig_maxiter,
        )
        if qaoa_mixer_norm is None
        else float(qaoa_mixer_norm)
    )

    alpha_grid, beta_grid = formulation4_alpha_beta_schedule_grid(N_steps)
    envelope_grid = auxiliary_envelope_grid(N_steps)
    delta_grid = auxiliary_schedule_grid(N_steps, rho)

    standard_flip_indices = make_flip_indices(n)
    state = plus_state(n) if initial_state is None else np.asarray(initial_state, dtype=complex).copy()

    energy_history = [expected_cost(state, cost_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]

    for step_index in range(1, int(N_steps) + 1):
        state, raw_norm = evolve_under_qaoa_mixer_assisted_interval_hamiltonian(
            state=state,
            cost_energies=cost_energies,
            standard_flip_indices=standard_flip_indices,
            qaoa_flip_data=qaoa_flip_data,
            alpha_left=float(alpha_grid[step_index - 1]),
            alpha_right=float(alpha_grid[step_index]),
            beta_left=float(beta_grid[step_index - 1]),
            beta_right=float(beta_grid[step_index]),
            delta_left=float(delta_grid[step_index - 1]),
            delta_right=float(delta_grid[step_index]),
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
        "delta_schedule": delta_grid[1:].copy(),
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "delta_grid": delta_grid,
        "auxiliary_envelope_grid": envelope_grid,
        "rho": float(rho),
        "standard_driver_spectral_norm": float(standard_norm),
        "qaoa_mixer_spectral_norm": float(qaoa_norm),
        "angle_scale": float(angle_scale),
        "qaoa_flip_data": qaoa_flip_data,
        "evolution_method": "continuous_ode_qaoa_mixer_assisted_baseline_schedule",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
        "eig_tol": float(eig_tol),
        "eig_maxiter": eig_maxiter,
    }


# summarize the auxiliary QAOA-style mixer for one graph.
def auxiliary_mixer_diagnostics(
    n: int,
    edges: set[tuple[int, int]],
    flip_data: list[dict],
    standard_driver_norm: float | None = None,
    qaoa_mixer_norm: float | None = None,
) -> dict:
    masks = non_neighbor_masks(n, edges)
    non_neighbor_counts = [int(mask.bit_count()) for mask in masks]
    allowed_counts = [int(data["allowed_count"]) for data in flip_data]

    standard_norm = (
        standard_driver_spectral_norm(n)
        if standard_driver_norm is None
        else float(standard_driver_norm)
    )
    qaoa_norm = (
        qaoa_mixer_spectral_norm(n, flip_data)
        if qaoa_mixer_norm is None
        else float(qaoa_mixer_norm)
    )

    return {
        "auxiliary_mixer_terms": int(n),
        "total_allowed_auxiliary_basis_flips": int(np.sum(allowed_counts)),
        "mean_allowed_auxiliary_basis_flips_per_qubit": float(np.mean(allowed_counts)) if allowed_counts else 0.0,
        "min_allowed_auxiliary_basis_flips_per_qubit": int(np.min(allowed_counts)) if allowed_counts else 0,
        "max_allowed_auxiliary_basis_flips_per_qubit": int(np.max(allowed_counts)) if allowed_counts else 0,
        "mean_non_neighbor_count": float(np.mean(non_neighbor_counts)) if non_neighbor_counts else 0.0,
        "min_non_neighbor_count": int(np.min(non_neighbor_counts)) if non_neighbor_counts else 0,
        "max_non_neighbor_count": int(np.max(non_neighbor_counts)) if non_neighbor_counts else 0,
        "standard_driver_spectral_norm": float(standard_norm),
        "qaoa_mixer_spectral_norm": float(qaoa_norm),
    }


# create one row per qubit describing the auxiliary mixer filter.
def auxiliary_mixer_specification_rows(
    instance_base: dict,
    n: int,
    edges: set[tuple[int, int]],
    flip_data: list[dict],
) -> list[dict]:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    rows = []

    for data in flip_data:
        qubit = int(data["qubit"])
        non_neighbors = [
            j
            for j in range(int(n))
            if j != qubit and tuple(sorted((qubit, j))) not in edge_set
        ]

        rows.append(
            {
                "instance_id": instance_base["instance_id"],
                "n": instance_base["n"],
                "p": instance_base["p"],
                "replicate": instance_base["replicate"],
                "qubit": qubit,
                "non_neighbor_count": int(len(non_neighbors)),
                "non_neighbors": " ".join(str(j) for j in non_neighbors),
                "allowed_auxiliary_basis_flips": int(data["allowed_count"]),
            }
        )

    return rows
