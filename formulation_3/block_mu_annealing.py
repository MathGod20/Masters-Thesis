from __future__ import annotations

from typing import Any
import time

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


# build adjacency lists for a simple undirected graph.
def adjacency_sets(n: int, edges: set[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(int(n))]

    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency


# deterministic greedy construction of independent blocks with size at most mu.
def greedy_independent_blocks_with_mu(
    n: int,
    edges: set[tuple[int, int]],
    mu: int,
    order: str = "largest_degree_first",
) -> dict[str, Any]:
    n = int(n)
    mu = int(mu)

    if mu < 1:
        raise ValueError("mu must be at least 1.")

    adjacency = adjacency_sets(n, edges)

    if order == "largest_degree_first":
        vertices = sorted(range(n), key=lambda i: (-len(adjacency[i]), i))
    elif order == "natural":
        vertices = list(range(n))
    else:
        raise ValueError(f"Unknown greedy block order: {order}")

    blocks: list[list[int]] = []

    for vertex in vertices:
        placed = False

        for block in blocks:
            if len(block) >= mu:
                continue

            # The block must remain an independent set in the original graph.
            if all(member not in adjacency[vertex] for member in block):
                block.append(vertex)
                placed = True
                break

        if not placed:
            blocks.append([vertex])

    # Sort vertices inside each block for deterministic output and easier reading.
    blocks = [sorted(block) for block in blocks]

    edge_set = {tuple(sorted(edge)) for edge in edges}
    for block in blocks:
        if len(block) > mu:
            raise RuntimeError("Greedy construction produced a block larger than mu.")

        for pos, i in enumerate(block):
            for j in block[pos + 1 :]:
                if tuple(sorted((i, j))) in edge_set:
                    raise RuntimeError("Greedy construction produced a non-independent block.")

    vertex_to_block = np.empty(n, dtype=int)
    for block_index, block in enumerate(blocks):
        for vertex in block:
            vertex_to_block[int(vertex)] = block_index

    return {
        "blocks": [tuple(block) for block in blocks],
        "vertex_to_block": vertex_to_block,
        "order": order,
        "mu": int(mu),
        "num_blocks": int(len(blocks)),
        "block_sizes": np.asarray([len(block) for block in blocks], dtype=int),
    }


# convert a basis-state index to a bit value for a given qubit.
def bit_value_from_index(index: np.ndarray | int, n: int, qubit: int) -> np.ndarray | int:
    shift = int(n) - 1 - int(qubit)
    return (np.asarray(index) >> shift) & 1


# precompute source indices and masks for the mu-block driver.
def precompute_block_mu_driver_data(
    n: int,
    blocks: list[tuple[int, ...]],
) -> dict[str, Any]:
    n = int(n)
    dim = 2**n
    indices = np.arange(dim, dtype=np.int64)
    flip_indices = make_flip_indices(n)

    vertex_to_block = np.empty(n, dtype=int)
    for block_index, block in enumerate(blocks):
        for vertex in block:
            vertex_to_block[int(vertex)] = block_index

    selected_count_by_block = []
    valid_block_mask = np.ones(dim, dtype=bool)

    for block in blocks:
        count = np.zeros(dim, dtype=np.int16)
        for vertex in block:
            count += bit_value_from_index(indices, n, int(vertex)).astype(np.int16)
        selected_count_by_block.append(count)
        valid_block_mask &= count <= 1

    add_sources: list[np.ndarray] = []
    remove_sources: list[np.ndarray] = []

    for qubit in range(n):
        block_index = int(vertex_to_block[qubit])
        count = selected_count_by_block[block_index]
        bit = bit_value_from_index(indices, n, qubit).astype(np.int8)

        # Add qubit i only from the empty state of its block.
        add_sources.append(indices[valid_block_mask & (count == 0) & (bit == 0)])

        # Remove qubit i only from a one-selected-vertex state in its block.
        remove_sources.append(indices[valid_block_mask & (count == 1) & (bit == 1)])

    return {
        "flip_indices": flip_indices,
        "add_sources": add_sources,
        "remove_sources": remove_sources,
        "valid_block_mask": valid_block_mask,
        "selected_count_by_block": selected_count_by_block,
        "vertex_to_block": vertex_to_block,
    }


# compute the closed-form ground state of the block driver.
def block_mu_driver_ground_state(
    n: int,
    blocks: list[tuple[int, ...]],
) -> np.ndarray:
    n = int(n)
    dim = 2**n
    state = np.zeros(dim, dtype=complex)

    for index in range(dim):
        amplitude = 1.0
        valid = True

        for block in blocks:
            selected = [
                vertex
                for vertex in block
                if int(bit_value_from_index(index, n, vertex)) == 1
            ]
            block_size = len(block)

            if len(selected) == 0:
                amplitude *= 1.0 / np.sqrt(2.0)
            elif len(selected) == 1:
                amplitude *= 1.0 / np.sqrt(2.0 * block_size)
            else:
                valid = False
                break

        if valid:
            state[index] = amplitude

    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise RuntimeError("Block-mu driver initial state has zero norm.")

    return state / norm


# return the analytic ground energy of the block driver.
def block_mu_driver_ground_energy(blocks: list[tuple[int, ...]]) -> float:
    return -float(sum(np.sqrt(len(block)) for block in blocks))


# compute H_D^mu psi.
def block_mu_driver_hamiltonian_times_state(
    state: np.ndarray,
    driver_data: dict[str, Any],
) -> np.ndarray:
    out = np.zeros_like(state, dtype=complex)
    flip_indices = driver_data["flip_indices"]

    for qubit, flipped in enumerate(flip_indices):
        add_sources = driver_data["add_sources"][qubit]
        remove_sources = driver_data["remove_sources"][qubit]

        if add_sources.size:
            out[flipped[add_sources]] -= state[add_sources]

        if remove_sources.size:
            out[flipped[remove_sources]] -= state[remove_sources]

    return out


# build the formulation-specific problem energies.
def build_block_mu_problem_energies(
    n: int,
    edges: set[tuple[int, int]],
    blocks: list[tuple[int, ...]],
    reward: float = 1.0,
    penalty: float = 2.0,
) -> np.ndarray:
    n = int(n)
    dim = 2**n
    indices = np.arange(dim, dtype=np.int64)

    vertex_to_block = np.empty(n, dtype=int)
    for block_index, block in enumerate(blocks):
        for vertex in block:
            vertex_to_block[int(vertex)] = block_index

    energies = np.zeros(dim, dtype=float)

    for i in range(n):
        bit_i = bit_value_from_index(indices, n, i).astype(float)
        energies -= float(reward) * bit_i

    for i, j in complement_edges(n, edges):
        # Within-block non-edge penalties are omitted because the driver and initial
        # state restrict the evolution to at most one selected vertex per block.
        if vertex_to_block[i] == vertex_to_block[j]:
            continue

        bit_i = bit_value_from_index(indices, n, i).astype(float)
        bit_j = bit_value_from_index(indices, n, j).astype(float)
        energies += float(penalty) * bit_i * bit_j

    return energies


# compute the probability assigned to states satisfying the block rule.
def block_valid_probability(state: np.ndarray, valid_block_mask: np.ndarray) -> float:
    probabilities = np.abs(state) ** 2
    return float(np.sum(probabilities[np.asarray(valid_block_mask, dtype=bool)]))


# compute H(t) psi for the formulation.
def block_mu_annealing_hamiltonian_times_state(
    state: np.ndarray,
    block_problem_energies: np.ndarray,
    driver_data: dict[str, Any],
    alpha: float,
    beta: float,
) -> np.ndarray:
    problem_part = float(alpha) * block_problem_energies * state
    driver_part = float(beta) * block_mu_driver_hamiltonian_times_state(state, driver_data)
    return problem_part + driver_part


# evolve one interval with linearly interpolated coefficients.
def evolve_under_block_mu_interval_hamiltonian(
    state: np.ndarray,
    block_problem_energies: np.ndarray,
    driver_data: dict[str, Any],
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

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        alpha_t = interpolate_endpoint_values(t, interval_length, alpha_left, alpha_right)
        beta_t = interpolate_endpoint_values(t, interval_length, beta_left, beta_right)

        hamiltonian_state = block_mu_annealing_hamiltonian_times_state(
            y,
            block_problem_energies,
            driver_data,
            alpha_t,
            beta_t,
        )

        return -1j * hamiltonian_state

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
        raise RuntimeError(f"Block-mu integration failed: {sol.message}")

    out = np.asarray(sol.y[:, -1], dtype=complex)
    raw_norm = float(np.linalg.norm(out))

    if normalize and raw_norm > 0.0:
        out = out / raw_norm

    return out, raw_norm


# simulate one mu value of the block-driver formulation.
def simulate_block_mu_annealing(
    n: int,
    edges: set[tuple[int, int]],
    evaluation_energies: np.ndarray,
    T: float,
    N_steps: int,
    mu: int,
    reward: float = 1.0,
    penalty: float = 2.0,
    block_order: str = "largest_degree_first",
    angle_scale: float = 1.0,
    ode_rtol: float = 1e-8,
    ode_atol: float = 1e-10,
    ode_method: str = "DOP853",
) -> dict[str, Any]:
    n = int(n)
    mu = int(mu)
    dt = float(T) / int(N_steps)
    effective_dt = float(angle_scale) * dt

    t0_block_construction = time.perf_counter()
    block_construction = greedy_independent_blocks_with_mu(
        n=n,
        edges=edges,
        mu=mu,
        order=block_order,
    )
    block_construction_time = time.perf_counter() - t0_block_construction

    blocks = block_construction["blocks"]

    t0_driver_precompute = time.perf_counter()
    driver_data = precompute_block_mu_driver_data(n, blocks)
    driver_precompute_time = time.perf_counter() - t0_driver_precompute

    t0_problem_energy = time.perf_counter()
    block_problem_energies = build_block_mu_problem_energies(
        n=n,
        edges=edges,
        blocks=blocks,
        reward=reward,
        penalty=penalty,
    )
    block_problem_energy_time = time.perf_counter() - t0_problem_energy

    t0_initial_state = time.perf_counter()
    state = block_mu_driver_ground_state(n, blocks)
    block_initial_state_time = time.perf_counter() - t0_initial_state
    alpha_grid, beta_grid = coefficient_schedule_grid(N_steps)

    energy_history = [expected_cost(state, evaluation_energies)]
    block_problem_energy_history = [expected_cost(state, block_problem_energies)]
    norm_history = [float(np.linalg.norm(state))]
    raw_norm_history = [float(np.linalg.norm(state))]
    block_valid_history = [block_valid_probability(state, driver_data["valid_block_mask"])]

    for step_index in range(1, int(N_steps) + 1):
        state, raw_norm = evolve_under_block_mu_interval_hamiltonian(
            state=state,
            block_problem_energies=block_problem_energies,
            driver_data=driver_data,
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

        energy_history.append(expected_cost(state, evaluation_energies))
        block_problem_energy_history.append(expected_cost(state, block_problem_energies))
        norm_history.append(float(np.linalg.norm(state)))
        raw_norm_history.append(raw_norm)
        block_valid_history.append(block_valid_probability(state, driver_data["valid_block_mask"]))

    return {
        "final_state": state,
        "energy_history": np.asarray(energy_history, dtype=float),
        "block_problem_energy_history": np.asarray(block_problem_energy_history, dtype=float),
        "norm_history": np.asarray(norm_history, dtype=float),
        "raw_norm_history": np.asarray(raw_norm_history, dtype=float),
        "block_valid_probability_history": np.asarray(block_valid_history, dtype=float),
        "dt": dt,
        "effective_dt": effective_dt,
        "alpha_schedule": alpha_grid[1:].copy(),
        "beta_schedule": beta_grid[1:].copy(),
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "angle_scale": float(angle_scale),
        "mu": int(mu),
        "block_problem_energies": block_problem_energies,
        "blocks": blocks,
        "block_sizes": block_construction["block_sizes"],
        "vertex_to_block": block_construction["vertex_to_block"],
        "block_order": block_order,
        "num_blocks": int(block_construction["num_blocks"]),
        "block_driver_ground_energy": block_mu_driver_ground_energy(blocks),
        "block_construction_time": float(block_construction_time),
        "driver_precompute_time": float(driver_precompute_time),
        "block_problem_energy_time": float(block_problem_energy_time),
        "block_initial_state_time": float(block_initial_state_time),
        "block_setup_time": float(
            block_construction_time
            + driver_precompute_time
            + block_problem_energy_time
            + block_initial_state_time
        ),
        "initial_state_type": "block_mu_driver_closed_form_ground_state",
        "driver_type": "block_mu_driver",
        "problem_type": "block_mu_reduced_max_clique_penalty",
        "evolution_method": "continuous_ode_linear_schedule",
        "ode_method": ode_method,
        "ode_rtol": float(ode_rtol),
        "ode_atol": float(ode_atol),
    }
