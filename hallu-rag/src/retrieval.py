"""
Retrieval module for dense, hybrid, and reranked retrieval.
Phase 3: Retrieve relevant chunks for a query.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

from .config import get_config
from .indexing import DenseIndex, BM25Index, load_indexes
from .utils import setup_logging


logger = setup_logging()


class Retriever:
    """Unified retriever supporting dense, hybrid, and reranked retrieval."""

    def __init__(
        self,
        strategy: str = "fixed",
        retrieval_type: str = "dense",
        use_rerank: bool = False,
        indexes: Optional[Dict[str, Any]] = None,
        chunks_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
    ):
        """
        Initialize retriever.

        Args:
            strategy: Chunking strategy ("fixed" or "semantic")
            retrieval_type: Retrieval type ("dense" or "hybrid")
            use_rerank: Whether to use cross-encoder reranking
            indexes: Pre-loaded indexes (optional)
            chunks_dir: Chunks directory (for loading chunks)
            index_dir: Index directory (for loading indexes)
        """
        self.strategy = strategy
        self.retrieval_type = retrieval_type
        self.use_rerank = use_rerank

        cfg = get_config()

        # Load indexes if not provided
        if indexes is None:
            self.indexes = load_indexes([strategy], index_dir)
        else:
            self.indexes = indexes

        if strategy not in self.indexes:
            raise ValueError(f"No indexes found for strategy: {strategy}")

        self.dense_index = self.indexes[strategy]["dense"]
        self.bm25_index = self.indexes[strategy]["bm25"]

        # Load embedding model for query encoding
        model_name = cfg.retrieval.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = cfg.get_device()
        self.embed_model = SentenceTransformer(model_name, device=device)

        # Load cross-encoder for reranking if needed
        self.cross_encoder = None
        if use_rerank:
            rerank_model = cfg.retrieval.get("rerank", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info(f"Loading cross-encoder: {rerank_model}")
            self.cross_encoder = CrossEncoder(rerank_model, device=device)

        # Load chunk texts for retrieval
        self.chunk_texts = self._load_chunk_texts(chunks_dir)

    def _load_chunk_texts(self, chunks_dir: Optional[Path]) -> Dict[str, str]:
        """Load chunk texts for retrieval results."""
        cfg = get_config()
        chunks_dir = Path(chunks_dir or cfg.data.get("chunks_dir", "data/chunks"))
        chunks_path = chunks_dir / f"chunks_{self.strategy}.jsonl"

        from .utils import load_jsonl
        chunks = load_jsonl(chunks_path)
        return {c["chunk_id"]: c["text"] for c in chunks}

    def retrieve_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Dense vector retrieval.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of result dictionaries with chunk_id, score, text
        """
        # Encode query
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Search
        scores, indices = self.dense_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.dense_index.chunk_ids):
                chunk_id = self.dense_index.chunk_ids[idx]
                results.append({
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "text": self.chunk_texts.get(chunk_id, ""),
                })

        return results

    def retrieve_bm25(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        BM25 sparse retrieval.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of result dictionaries
        """
        scores, indices = self.bm25_index.search(query, top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.bm25_index.chunk_ids):
                chunk_id = self.bm25_index.chunk_ids[idx]
                results.append({
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "text": self.chunk_texts.get(chunk_id, ""),
                })

        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining BM25 and dense scores.

        Args:
            query: Query string
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores

        Returns:
            List of result dictionaries
        """
        # Get more candidates from each
        candidate_k = top_k * 3

        dense_results = self.retrieve_dense(query, candidate_k)
        bm25_results = self.retrieve_bm25(query, candidate_k)

        # Create score maps
        dense_scores = {r["chunk_id"]: r["score"] for r in dense_results}
        bm25_scores = {r["chunk_id"]: r["score"] for r in bm25_results}

        # Normalize scores to [0, 1]
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            values = list(scores_dict.values())
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores)

        # Combine scores
        all_chunk_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
        combined = []

        for chunk_id in all_chunk_ids:
            d_score = dense_norm.get(chunk_id, 0)
            b_score = bm25_norm.get(chunk_id, 0)
            combined_score = dense_weight * d_score + bm25_weight * b_score
            combined.append({
                "chunk_id": chunk_id,
                "score": combined_score,
                "text": self.chunk_texts.get(chunk_id, ""),
            })

        # Sort and return top-k
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query string
            candidates: Candidate results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if self.cross_encoder is None or not candidates:
            return candidates[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Get scores
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        # Add scores and sort
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Main retrieval method.

        Args:
            query: Query string
            top_k: Number of results (from config if None)

        Returns:
            List of retrieved chunks with scores and text
        """
        cfg = get_config()

        if top_k is None:
            if self.retrieval_type == "dense":
                top_k = cfg.retrieval.get("dense", {}).get("top_k", 20)
            else:
                top_k = cfg.retrieval.get("hybrid", {}).get("top_k", 20)

        # Initial retrieval
        if self.retrieval_type == "dense":
            candidates = self.retrieve_dense(query, top_k * 3 if self.use_rerank else top_k)
        elif self.retrieval_type == "hybrid":
            bm25_w = cfg.retrieval.get("hybrid", {}).get("bm25_weight", 0.5)
            dense_w = cfg.retrieval.get("hybrid", {}).get("dense_weight", 0.5)
            candidates = self.retrieve_hybrid(query, top_k * 3 if self.use_rerank else top_k, bm25_w, dense_w)
        else:
            raise ValueError(f"Unknown retrieval type: {self.retrieval_type}")

        # Rerank if enabled
        if self.use_rerank and candidates:
            rerank_k = cfg.retrieval.get("rerank", {}).get("top_k", 5)
            candidates = self.rerank(query, candidates, rerank_k)
        else:
            candidates = candidates[:top_k]

        # Add citation indices
        for i, c in enumerate(candidates):
            c["citation_id"] = f"C{i+1}"

        return candidates


def create_retriever(
    strategy: str = "fixed",
    retrieval_type: str = "dense",
    use_rerank: bool = False,
    indexes: Optional[Dict[str, Any]] = None,
    chunks_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> Retriever:
    """
    Factory function to create a retriever.

    Args:
        strategy: Chunking strategy
        retrieval_type: Retrieval type
        use_rerank: Whether to use reranking
        indexes: Pre-loaded indexes
        chunks_dir: Chunks directory
        index_dir: Index directory

    Returns:
        Configured Retriever instance
    """
    return Retriever(
        strategy=strategy,
        retrieval_type=retrieval_type,
        use_rerank=use_rerank,
        indexes=indexes,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Retrieval module loaded successfully")
    print(f"Default strategy: fixed")
    print(f"Default retrieval: dense")