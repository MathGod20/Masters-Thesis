from pathlib import Path
import hashlib
import random
from .graph_io import write_edgelist


# create a reproducible seed for one instance.
def stable_seed(base_seed: int, n: int, p: float, replicate: int) -> int:
    text = f"{base_seed}|ER|{n}|{p:.6f}|{replicate}"

    # hash the instance settings into a fixed integer.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # convert part of the hash to a seed accepted by random.Random.
    return int(digest[:16], 16) % (2**32)


# generate edges for an Erdős-Rényi graph.
def generate_er_edges(n: int, p: float, seed: int) -> set[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()

    for i in range(n):
        for j in range(i + 1, n):
            # add each possible edge with probability p.
            if rng.random() < p:
                edges.add((i, j))

    return edges


# generate and save all ER instances in the experiment grid.
def generate_instance_grid(
    output_dir: str | Path,
    n_values,
    p_values,
    replicates: int,
    base_seed: int = 2026,
) -> list[Path]:
    output_dir = Path(output_dir)

    # create the instance folder.
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

    for n in n_values:
        for p in p_values:
            # encode p in the filename, for example p=0.10 becomes p010.
            p_tag = int(round(100 * p))

            for r in range(replicates):
                # each instance gets its own deterministic seed.
                seed = stable_seed(base_seed, int(n), float(p), int(r))

                edges = generate_er_edges(int(n), float(p), seed)

                path = output_dir / f"ER_n{int(n):03d}_p{p_tag:03d}_r{int(r):02d}.edgelist"

                write_edgelist(path, int(n), edges)
                paths.append(path)

    return paths