import numpy as np
from typing import Dict, Tuple, Set

# qubo as a dictionary of coefficient pairs.
Qubo = Dict[Tuple[int, int], float]


# clean and combine qubo coefficients.
def normalize_qubo(Q: Qubo) -> Qubo:
    Q_norm: Qubo = {}

    for (i, j), value in Q.items():
        # remove numerical zero terms.
        if abs(value) < 1e-15:
            continue

        # store every pair in canonical order.
        a, b = sorted((int(i), int(j)))

        # combine coefficients that refer to the same pair.
        Q_norm[(a, b)] = Q_norm.get((a, b), 0.0) + float(value)

    # remove terms that became zero after combining.
    return {key: value for key, value in Q_norm.items() if abs(value) > 1e-15}


# build the maximum clique qubo.
def build_max_clique_qubo(
    n: int,
    edges: Set[Tuple[int, int]],
    reward: float = 1.0,
    penalty: float = 2.0,
) -> Qubo:
    # store graph edges in canonical order.
    edges = {tuple(sorted(edge)) for edge in edges}

    # reward selecting each node.
    Q: Qubo = {(i, i): -float(reward) for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            # penalize selecting two nodes that are not connected.
            if (i, j) not in edges:
                Q[(i, j)] = Q.get((i, j), 0.0) + float(penalty)

    return normalize_qubo(Q)


# convert a basis-state index to a bitstring.
def bitstring_from_index(index: int, n: int) -> tuple[int, ...]:
    return tuple((index >> (n - 1 - qubit)) & 1 for qubit in range(n))


# convert a bitstring tuple to a string.
def bitstring_to_str(bits) -> str:
    return "".join(str(int(bit)) for bit in bits)


# compute the qubo energy of one bitstring.
def qubo_energy(bits, Q: Qubo) -> float:
    energy = 0.0

    for (i, j), value in Q.items():
        # linear term.
        if i == j:
            energy += value * bits[i]

        # quadratic term.
        else:
            energy += value * bits[i] * bits[j]

    return float(energy)


# compute the qubo energy of every computational basis state.
def build_cost_energies(n: int, Q: Qubo) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    energies = np.zeros(2**n, dtype=float)
    bitstrings: list[tuple[int, ...]] = []

    for index in range(2**n):
        bits = bitstring_from_index(index, n)
        bitstrings.append(bits)
        energies[index] = qubo_energy(bits, Q)

    return energies, bitstrings


# convert the qubo dictionary to a dense matrix.
def qubo_to_dense_matrix(n: int, Q: Qubo) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)

    for (i, j), value in Q.items():
        matrix[i, j] += float(value)

    return matrix