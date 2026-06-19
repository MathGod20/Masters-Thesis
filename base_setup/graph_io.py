from pathlib import Path
from typing import Iterable, Set, Tuple

# edge as a pair of node indices.
Edge = Tuple[int, int]


# clean edges for a simple undirected graph.
def normalize_edges(edges: Iterable[Edge]) -> Set[Edge]:
    clean_edges: Set[Edge] = set()

    for u, v in edges:
        # skip self-loops.
        if u == v:
            continue

        # use canonical edge order.
        if u > v:
            u, v = v, u

        # remove duplicates.
        clean_edges.add((int(u), int(v)))

    return clean_edges


# read an edgelist file.
def read_edgelist(path: str | Path) -> tuple[int, Set[Edge]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()

        # expect header: n m.
        if len(first) != 2:
            raise ValueError(f"Invalid first line in {path}: expected 'n m'.")

        n, m = map(int, first)
        edges = []

        for line_number, line in enumerate(f, start=2):
            # skip empty lines.
            if not line.strip():
                continue

            values = line.split()

            # each edge must have two endpoints.
            if len(values) != 2:
                raise ValueError(f"Invalid edge on line {line_number} in {path}.")

            edges.append(tuple(map(int, values)))

    clean_edges = normalize_edges(edges)

    # validate edge count.
    if len(clean_edges) != m:
        raise ValueError(
            f"Edge count mismatch in {path}: header has {m}, "
            f"file has {len(clean_edges)} unique edges."
        )

    return n, clean_edges


# write an edgelist file.
def write_edgelist(path: str | Path, n: int, edges: Iterable[Edge]) -> None:
    path = Path(path)

    # create parent folder.
    path.parent.mkdir(parents=True, exist_ok=True)

    # deterministic file order.
    clean_edges = sorted(normalize_edges(edges))

    with path.open("w", encoding="utf-8") as f:
        # write header: n m.
        f.write(f"{n} {len(clean_edges)}\n")

        for u, v in clean_edges:
            f.write(f"{u} {v}\n")


# extract instance information from the filename.
def parse_instance_name(path: str | Path) -> dict[str, int | float | str]:
    stem = Path(path).stem
    parts = stem.split("_")

    # fallback for unexpected names.
    if len(parts) < 4:
        return {"instance_id": stem}

    try:
        # example: ER_n015_p010_r00.
        n = int(parts[1][1:])
        p = int(parts[2][1:]) / 100.0
        r = int(parts[3][1:])

        return {"instance_id": stem, "n": n, "p": p, "replicate": r}

    except Exception:
        # fallback if parsing fails.
        return {"instance_id": stem}