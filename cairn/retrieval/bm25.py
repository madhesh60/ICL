"""BM25: the classic "which documents share rare words with my question" score.

Small enough to own.  Reading it once teaches you what every search engine did
before neural networks, and why it is still a strong baseline:

    score(doc) = sum over query words of   idf(word) * tf_part(word, doc)

    idf      - rare words count more ("tiramisu" beats "the")
    tf_part  - a word appearing 3 times counts more than once, but not 3x more,
               and long documents do not win just by being long
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np


class BM25Index:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self._avg_len = float(lengths.mean()) if self.n_docs else 0.0

        postings: dict[str, tuple[list[int], list[int]]] = {}
        for doc_id, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                ids, tfs = postings.setdefault(term, ([], []))
                ids.append(doc_id)
                tfs.append(tf)

        # For each word keep (which docs contain it, the ready-to-add score for each doc).
        # All the arithmetic is done here once, so a query is only a few array additions.
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        norm = k1 * (1.0 - b + b * lengths / max(self._avg_len, 1e-9))
        for term, (ids, tfs) in postings.items():
            doc_ids = np.asarray(ids, dtype=np.int32)
            tf = np.asarray(tfs, dtype=np.float32)
            df = len(ids)
            idf = float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))
            self._postings[term] = (doc_ids, idf * tf * (k1 + 1.0) / (tf + norm[doc_ids]))

    def idf(self, term: str) -> float:
        """Rarity weight of a word; 0.0 for words we never saw."""
        entry = self._postings.get(term)
        if entry is None:
            return 0.0
        df = len(entry[0])
        return float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))

    def scores(self, query: Iterable[str]) -> np.ndarray:
        """BM25 score of every document for this query (zeros for no overlap)."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(query):
            entry = self._postings.get(term)
            if entry is not None:
                out[entry[0]] += entry[1]
        return out

    def top_k(
        self, query: Iterable[str], k: int, mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """The k best documents as (index, score), best first.

        `mask` is a True/False array: only documents marked True may be returned.
        Ties are broken by document index so results never depend on luck.
        """
        scores = self.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        order = np.lexsort((np.arange(self.n_docs), -scores))[:k]
        return [(int(i), float(scores[i])) for i in order]
