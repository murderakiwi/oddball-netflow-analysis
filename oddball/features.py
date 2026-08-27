"""Egonet feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


class Graph:
    def __init__(self, edges: pd.DataFrame, weight_col: str):
        src = edges["src"].to_numpy()
        dst = edges["dst"].to_numpy()
        w = edges[weight_col].to_numpy(dtype=float)

        nodes = pd.unique(np.concatenate([src, dst]))
        self.nodes = nodes
        self.index = {n: i for i, n in enumerate(nodes)}
        self.n = len(nodes)

        si = np.array([self.index[x] for x in src], dtype=np.int64)
        di = np.array([self.index[x] for x in dst], dtype=np.int64)
        self.ei, self.ej, self.ew = si, di, w

        self.adj: list[set] = [set() for _ in range(self.n)]
        self.wmap: dict[tuple[int, int], float] = {}
        for u, v, ww in zip(si, di, w):
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.wmap[(u, v) if u < v else (v, u)] = ww

    def weight(self, u: int, v: int) -> float:
        return self.wmap.get((u, v) if u < v else (v, u), 0.0)


def _weighted_egonet_eigenvalues(
    g: Graph, E: np.ndarray, dense_limit: int = 96
) -> np.ndarray:
    out = np.zeros(g.n, dtype=float)
    for ego in range(g.n):
        neighbours = g.adj[ego]
        degree = len(neighbours)
        if degree == 0:
            continue
        incident = np.fromiter(
            (g.weight(ego, v) for v in neighbours), dtype=float, count=degree
        )
        if E[ego] == degree:
            out[ego] = float(np.sqrt(np.dot(incident, incident)))
            continue

        vertices = [ego, *neighbours]
        local = {v: j for j, v in enumerate(vertices)}
        rows, cols, values = [], [], []

        for v, weight in zip(neighbours, incident):
            lu, lv = local[ego], local[v]
            rows.extend((lu, lv))
            cols.extend((lv, lu))
            values.extend((weight, weight))
        for u in neighbours:
            for v in g.adj[u] & neighbours:
                if local[u] < local[v]:
                    lu, lv = local[u], local[v]
                    weight = g.weight(u, v)
                    rows.extend((lu, lv))
                    cols.extend((lv, lu))
                    values.extend((weight, weight))

        size = len(vertices)
        if size <= dense_limit:
            matrix = np.zeros((size, size), dtype=float)
            matrix[rows, cols] = values
            out[ego] = float(np.linalg.eigvalsh(matrix)[-1])
        else:
            matrix = csr_matrix((values, (rows, cols)), shape=(size, size))
            try:
                out[ego] = float(
                    eigsh(matrix, k=1, which="LA", return_eigenvectors=False, tol=1e-7)[
                        0
                    ]
                )
            except ArpackNoConvergence:
                out[ego] = float(np.linalg.eigvalsh(matrix.toarray())[-1])
    return out


def egonet_features(
    g: Graph, n_as_degree: bool = True, include_eigenvalue: bool = False
) -> pd.DataFrame:
    n = g.n
    N = np.zeros(n, dtype=np.float64) if n_as_degree else np.ones(n, dtype=np.float64)
    E = np.zeros(n, dtype=np.float64)
    W = np.zeros(n, dtype=np.float64)

    for i in range(n):
        N[i] = len(g.adj[i]) if n_as_degree else 1.0 + len(g.adj[i])

    np.add.at(E, g.ei, 1.0)
    np.add.at(E, g.ej, 1.0)
    np.add.at(W, g.ei, g.ew)
    np.add.at(W, g.ej, g.ew)

    for u, v, ww in zip(g.ei, g.ej, g.ew):
        au, av = g.adj[u], g.adj[v]
        if len(au) > len(av):
            au, av = av, au
        for c in au:
            if c in av:
                E[c] += 1.0
                W[c] += ww

    result = pd.DataFrame({"node": g.nodes, "N": N, "E": E, "W": W})
    if include_eigenvalue:
        result["lambda_w"] = _weighted_egonet_eigenvalues(g, E)
    return result
