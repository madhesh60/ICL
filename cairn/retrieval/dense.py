"""Small fixed-size vectors for text, and grouping similar texts together.

BM25 answers "which examples match this query".  For grouping ("which queries
are alike?") we want every text as a point in the same space.  We get that with
the hashing trick: each feature is dropped into one of `dim` buckets by a hash,
no vocabulary file needed.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    # zlib.crc32 gives the same number on every run.  Python's built-in hash()
    # does NOT (it is salted per process), which would silently change clusters
    # between runs and break reproducibility.
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, 1.0 if (h >> 31) & 1 else -1.0


class HashedTfidf:
    """TF-IDF vectors of fixed width, scaled to length 1."""

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self._idf: np.ndarray | None = None

    def _raw(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        matrix = np.zeros((len(docs), self.dim), dtype=np.float32)
        for row, doc in enumerate(docs):
            for feature in doc:
                col, sign = _bucket(feature, self.dim)
                matrix[row, col] += sign
        return matrix

    def fit(self, docs: Sequence[Sequence[str]]) -> HashedTfidf:
        present = (self._raw(docs) != 0).sum(axis=0)
        self._idf = (np.log((1.0 + len(docs)) / (1.0 + present)) + 1.0).astype(np.float32)
        return self

    def transform(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("call fit() before transform()")
        matrix = self._raw(docs)
        # Dampen repeats (a word used 9 times is not 9 times as important).
        matrix = np.sign(matrix) * np.log1p(np.abs(matrix)) * self._idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-9)


def spherical_kmeans(
    vectors: np.ndarray, k: int, seed: int = 0, n_iter: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Group unit vectors into k clusters by direction.  Returns (centroids, labels).

    Starting points are chosen "k-means++" style: each next start is picked far
    from the ones already chosen, which avoids two starts landing in one clump.
    """
    n = vectors.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    centroids[0] = vectors[rng.integers(n)]
    closest = 1.0 - vectors @ centroids[0]
    for j in range(1, k):
        weights = np.maximum(closest, 0.0) ** 2
        total = float(weights.sum())
        pick = int(rng.choice(n, p=weights / total)) if total > 0 else int(rng.integers(n))
        centroids[j] = vectors[pick]
        closest = np.minimum(closest, 1.0 - vectors @ centroids[j])

    labels = np.zeros(n, dtype=np.int32)
    for step in range(n_iter):
        new_labels = np.argmax(vectors @ centroids.T, axis=1).astype(np.int32)
        if step > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = vectors[labels == j]
            if len(members) == 0:
                # An empty cluster is re-seeded with the point that fits worst anywhere.
                worst = int(np.argmin(np.max(vectors @ centroids.T, axis=1)))
                centroids[j] = vectors[worst]
                continue
            mean = members.sum(axis=0)
            centroids[j] = mean / max(float(np.linalg.norm(mean)), 1e-9)
    return centroids, labels
