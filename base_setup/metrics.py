import numpy as np


# check whether a bitstring represents a clique.
def is_clique(bits: tuple[int, ...], edges: set[tuple[int, int]]) -> bool:
    edge_set = {tuple(sorted(edge)) for edge in edges}
    selected = [i for i, bit in enumerate(bits) if bit == 1]

    return all(
        tuple(sorted((u, v))) in edge_set
        for idx, u in enumerate(selected)
        for v in selected[idx + 1:]
    )


# mark which bitstrings are feasible cliques.
def feasibility_mask(
    bitstrings: list[tuple[int, ...]],
    edges: set[tuple[int, int]],
) -> np.ndarray:
    return np.asarray([is_clique(bits, edges) for bits in bitstrings], dtype=bool)


# compute the variance of the full energy distribution.
def energy_variance_from_probabilities(
    probabilities: np.ndarray,
    cost_energies: np.ndarray,
    expected_energy: float,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    cost_energies = np.asarray(cost_energies, dtype=float)

    deviations = cost_energies - float(expected_energy)
    return float(np.dot(probabilities, deviations**2))


# compute the expected energy conditioned on feasible bitstrings.
def expected_feasible_energy_from_probabilities(
    probabilities: np.ndarray,
    cost_energies: np.ndarray,
    feasible: np.ndarray,
    probability_tol: float = 1e-15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    cost_energies = np.asarray(cost_energies, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)

    feasible_probability = float(np.sum(probabilities[feasible]))

    # return nan if feasible states have essentially no probability.
    if feasible_probability <= float(probability_tol):
        return float("nan")

    feasible_energy = float(
        np.dot(probabilities[feasible], cost_energies[feasible])
    )

    return float(feasible_energy / feasible_probability)


# compute the energy variance conditioned on feasible bitstrings.
def feasible_energy_variance_from_probabilities(
    probabilities: np.ndarray,
    cost_energies: np.ndarray,
    feasible: np.ndarray,
    expected_feasible_energy: float,
    probability_tol: float = 1e-15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    cost_energies = np.asarray(cost_energies, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)

    feasible_probability = float(np.sum(probabilities[feasible]))

    # return nan if feasible states have essentially no probability.
    if feasible_probability <= float(probability_tol):
        return float("nan")

    deviations = cost_energies[feasible] - float(expected_feasible_energy)
    feasible_variance = np.dot(probabilities[feasible], deviations**2)

    return float(feasible_variance / feasible_probability)


# compute the main metrics of the final state.
def final_state_metrics(
    state: np.ndarray,
    cost_energies: np.ndarray,
    bitstrings,
    edges,
) -> dict:
    probabilities = np.abs(state) ** 2

    ground_energy = float(np.min(cost_energies))
    ground_indices = np.where(np.isclose(cost_energies, ground_energy))[0]

    feasible = feasibility_mask(bitstrings, edges)

    p_ground = float(np.sum(probabilities[ground_indices]))
    p_feas = float(np.sum(probabilities[feasible]))

    final_expected_energy = float(np.dot(probabilities, cost_energies))

    energy_variance = energy_variance_from_probabilities(
        probabilities=probabilities,
        cost_energies=cost_energies,
        expected_energy=final_expected_energy,
    )

    expected_feasible_energy = expected_feasible_energy_from_probabilities(
        probabilities=probabilities,
        cost_energies=cost_energies,
        feasible=feasible,
    )

    feasible_energy_variance = feasible_energy_variance_from_probabilities(
        probabilities=probabilities,
        cost_energies=cost_energies,
        feasible=feasible,
        expected_feasible_energy=expected_feasible_energy,
    )

    return {
        "final_expected_energy": final_expected_energy,
        "energy_variance": energy_variance,
        "ground_energy": ground_energy,
        "ground_state_probability": p_ground,
        "expected_feasible_energy": expected_feasible_energy,
        "feasible_energy_variance": feasible_energy_variance,
        "feasibility_probability": p_feas,
    }