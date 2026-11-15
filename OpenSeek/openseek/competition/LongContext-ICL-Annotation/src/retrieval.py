# retrieval.py
"""
Smart example picker using semantic retrieval with FAISS.
This module implements retrieval-based in-context learning (ICL) for optimal example selection.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional


class SemanticRetriever:
    """
    Semantic retrieval system using FAISS for finding the most similar examples.
    Implements the winning strategy: retrieval-based ICL with diversity sampling.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic retriever with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """
        self.embedder = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.demo_set: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, demo_set: List[Dict[str, Any]], show_progress: bool = True) -> None:
        """
        Build a FAISS index from demo examples.

        Args:
            demo_set: List of example dicts with 'input' and 'label' keys
            show_progress: Whether to show progress bar during encoding
        """
        self.demo_set = demo_set
        demo_texts = [d['input'] for d in demo_set]

        print(f"Building retrieval index with {len(demo_set)} examples...")
        self.embeddings = self.embedder.encode(
            demo_texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        self.embeddings = np.array(self.embeddings).astype('float32')

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        print(f"Index built successfully with {self.index.ntotal} examples.")

    def get_top_k_similar(self, test_input: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Find the k most similar demo examples to the test input.

        Args:
            test_input: The input text to find similar examples for
            k: Number of similar examples to retrieve

        Returns:
            List of k most similar demo examples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query = self.embedder.encode([test_input]).astype('float32')
        distances, indices = self.index.search(query, k)
        return [self.demo_set[i] for i in indices[0]]

    def get_diverse_demos(
        self,
        test_input: str,
        k_similar: int = 4,
        k_diverse: int = 4
    ) -> List[Dict[str, Any]]:
        """
        WINNING TRICK: Get diverse examples combining similarity and label coverage.

        This strategy ensures:
        1. Most similar examples are included (k_similar)
        2. Examples covering different label types are included (k_diverse)

        This prevents the model from getting one-sided examples and improves accuracy.

        Args:
            test_input: The input text to find examples for
            k_similar: Number of similar examples to include
            k_diverse: Number of diverse examples (one per label type) to include

        Returns:
            Combined list of similar and diverse examples
        """
        # Get k_similar most similar examples
        similar = self.get_top_k_similar(test_input, k=k_similar)
        similar_inputs = set(d['input'] for d in similar)

        # Get one example per unique label type (for diversity)
        seen_labels = set()
        diverse = []
        for demo in self.demo_set:
            label = demo.get('label') or demo.get('output', [None])[0]
            if label not in seen_labels and demo['input'] not in similar_inputs:
                diverse.append(demo)
                seen_labels.add(label)
            if len(diverse) >= k_diverse:
                break

        # Combine: similar first, then diverse ones
        return similar + diverse

    def get_balanced_examples(
        self,
        test_input: str,
        k: int = 8,
        balance_labels: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get balanced examples with optional label balancing.

        Args:
            test_input: The input text to find examples for
            k: Total number of examples to retrieve
            balance_labels: Whether to balance examples across different labels

        Returns:
            List of balanced example selections
        """
        if not balance_labels:
            return self.get_top_k_similar(test_input, k=k)

        # Get more than needed and balance
        all_similar = self.get_top_k_similar(test_input, k=k * 2)

        if len(all_similar) <= k:
            return all_similar

        # Group by label
        by_label: Dict[str, List[Dict]] = {}
        for demo in all_similar:
            label = demo.get('label') or demo.get('output', [None])[0]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(demo)

        # Distribute examples evenly across labels
        result = []
        max_per_label = max(1, k // len(by_label))

        for label, demos in by_label.items():
            result.extend(demos[:max_per_label])

        # Fill remaining slots with most similar
        remaining = k - len(result)
        if remaining > 0:
            for demo in all_similar:
                if len(result) >= k:
                    break
                if demo not in result:
                    result.append(demo)

        return result[:k]


def build_index(demo_set: List[Dict[str, Any]], model_name: str = 'all-MiniLM-L6-v2') -> tuple:
    """
    Standalone function to build a retrieval index.

    Args:
        demo_set: List of example dicts with 'input' and 'label' keys
        model_name: Name of the sentence transformer model

    Returns:
        Tuple of (faiss_index, embeddings)
    """
    embedder = SentenceTransformer(model_name)
    demo_texts = [d['input'] for d in demo_set]

    print(f"Building retrieval index with {len(demo_set)} examples...")
    embeddings = embedder.encode(demo_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Index built successfully with {index.ntotal} examples.")
    return index, embeddings


def get_top_k_similar(
    test_input: str,
    demo_set: List[Dict[str, Any]],
    index: faiss.IndexFlatL2,
    k: int = 4,
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[Dict[str, Any]]:
    """
    Standalone function to get top k similar examples.

    Args:
        test_input: Input text to find similar examples for
        demo_set: List of all demo examples
        index: FAISS index
        k: Number of examples to retrieve
        model_name: Name of sentence transformer model

    Returns:
        List of k most similar examples
    """
    embedder = SentenceTransformer(model_name)
    query = embedder.encode([test_input]).astype('float32')
    distances, indices = index.search(query, k)
    return [demo_set[i] for i in indices[0]]


def get_diverse_demos(
    test_input: str,
    demo_set: List[Dict[str, Any]],
    index: faiss.IndexFlatL2,
    k_similar: int = 4,
    k_diverse: int = 4,
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[Dict[str, Any]]:
    """
    Standalone function to get diverse examples combining similarity and label coverage.

    Args:
        test_input: Input text to find examples for
        demo_set: List of all demo examples
        index: FAISS index
        k_similar: Number of similar examples
        k_diverse: Number of diverse examples
        model_name: Name of sentence transformer model

    Returns:
        Combined list of similar and diverse examples
    """
    embedder = SentenceTransformer(model_name)

    # Get k_similar most similar examples
    similar = get_top_k_similar(test_input, demo_set, index, k=k_similar, model_name=model_name)
    similar_inputs = set(d['input'] for d in similar)

    # Get one example per unique label type
    seen_labels = set()
    diverse = []
    for demo in demo_set:
        label = demo.get('label') or demo.get('output', [None])[0]
        if label not in seen_labels and demo['input'] not in similar_inputs:
            diverse.append(demo)
            seen_labels.add(label)
        if len(diverse) >= k_diverse:
            break

    return similar + diverse
