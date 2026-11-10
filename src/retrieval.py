import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from src.indexing import IndexBundle, dense_search, sparse_search
from src.config import load_config


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    chunks: List[Dict]
    reranked: bool
    candidate_count: int
    final_count: int
    timings: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


class Retriever:
    """Unified retriever supporting dense, sparse, hybrid, and reranked search."""

    def __init__(self, bundle: IndexBundle, cfg: dict, reranker_model: Optional[str] = None):
        self.bundle = bundle
        self.cfg = cfg
        self.reranker_model = reranker_model or cfg.get("reranker", {}).get("model")
        self._cross_encoder = None
        self.rrf_k = cfg.get("retrieval", {}).get("rrf_k", 60)
        self.candidate_k = cfg.get("retrieval", {}).get("candidate_k", 20)
        self.top_k = cfg.get("retrieval", {}).get("top_k", 5)
        self._last_stage_timings = {}

    @property
    def cross_encoder(self) -> CrossEncoder:
        """Lazy load cross-encoder."""
        if self._cross_encoder is None and self.reranker_model:
            self._cross_encoder = CrossEncoder(self.reranker_model)
        return self._cross_encoder

    def dense(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Dense search via FAISS. Returns list of (chunk_id, score)."""
        started = time.perf_counter()
        results = dense_search(self.bundle, query, k)
        self._last_stage_timings["dense"] = time.perf_counter() - started
        self._last_stage_timings.update(self.bundle.search_timings or {})
        return [(chunk["chunk_id"], score) for chunk, score in results]

    def bm25(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Sparse search via BM25. Returns list of (chunk_id, score)."""
        started = time.perf_counter()
        results = sparse_search(self.bundle, query, k)
        self._last_stage_timings["bm25"] = time.perf_counter() - started
        return [(chunk["chunk_id"], score) for chunk, score in results]

    def hybrid(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Hybrid search via RRF fusion of dense and BM25."""
        candidate_k = self.candidate_k

        # Get dense results
        dense_results = self.dense(query, candidate_k)
        dense_rank = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(dense_results)}

        # Get BM25 results
        bm25_results = self.bm25(query, candidate_k)
        bm25_rank = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(bm25_results)}

        # RRF fusion: score = sum(1 / (rrf_k + rank))
        all_chunk_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        rrf_scores = {}

        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[chunk_id])
            if chunk_id in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score descending
        fusion_started = time.perf_counter()
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        self._last_stage_timings["fusion"] = time.perf_counter() - fusion_started
        return sorted_results[:k]

    def rerank(self, query: str, candidates: List[str], k: int) -> Tuple[List[Tuple[str, float]], float]:
        """Rerank candidates using cross-encoder. Returns list of (chunk_id, score) and time taken."""
        if not candidates:
            return [], 0.0

        # Prepare pairs for cross-encoder
        pairs = []
        chunk_id_to_text = {chunk["chunk_id"]: chunk for chunk in self.bundle.chunks}

        for chunk_id in candidates:
            if chunk_id in chunk_id_to_text:
                text = chunk_id_to_text[chunk_id]["text"]
                pairs.append((query, text))
            else:
                pairs.append((query, ""))

        if not self.cross_encoder:
            # No reranker available, return original order with dummy scores
            return [(cid, 1.0 - i * 0.01) for i, cid in enumerate(candidates[:k])], 0.0

        # Get cross-encoder scores with timing
        start = time.perf_counter()
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        rerank_time = time.perf_counter() - start

        # Pair with chunk_ids and sort
        ranked = list(zip(candidates, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)

        final_chunks = ranked[:k]
        return final_chunks, rerank_time

    def retrieve(
        self,
        query: str,
        mode: str = "dense",
        use_rerank: bool = False,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Unified retrieval interface.

        Args:
            query: Search query
            mode: "dense", "bm25", or "hybrid"
            use_rerank: Whether to apply cross-encoder reranking
            top_k: Number of final results (defaults to config value)

        Returns:
            RetrievalResult with chunks, rerank flag, counts, and timings
        """
        timings = {}
        top_k = top_k or self.top_k
        candidate_k = self.candidate_k if use_rerank else top_k

        # Initial retrieval with timing
        start = time.perf_counter()
        if mode == "dense":
            stage_start = time.perf_counter()
            candidates = self.dense(query, candidate_k)
            timings.update(self._last_stage_timings)
        elif mode == "bm25":
            stage_start = time.perf_counter()
            candidates = self.bm25(query, candidate_k)
            timings.update(self._last_stage_timings)
        elif mode == "hybrid":
            stage_start = time.perf_counter()
            candidates = self.hybrid(query, candidate_k)
            timings.update(self._last_stage_timings)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        timings["initial_retrieval"] = time.perf_counter() - start

        candidate_ids = [cid for cid, _ in candidates]
        candidate_count = len(candidate_ids)

        # Rerank if requested
        reranked = False
        rerank_time = 0.0
        if use_rerank and self.reranker_model:
            reranked_results, rerank_time = self.rerank(query, candidate_ids, top_k)
            final_ids = [cid for cid, _ in reranked_results]
            final_scores = {cid: score for cid, score in reranked_results}
            reranked = True
        else:
            final_ids = candidate_ids[:top_k]
            final_scores = {cid: score for cid, score in candidates[:top_k]}

        timings["rerank"] = rerank_time

        # Build result dicts with full chunk info
        chunk_id_to_chunk = {chunk["chunk_id"]: chunk for chunk in self.bundle.chunks}
        results = []

        for rank, chunk_id in enumerate(final_ids, 1):
            if chunk_id in chunk_id_to_chunk:
                chunk = chunk_id_to_chunk[chunk_id].copy()
                chunk["score"] = final_scores.get(chunk_id, 0.0)
                chunk["rank"] = rank
                # Add score type label
                if reranked:
                    chunk["score_type"] = "rerank"
                elif mode == "dense":
                    chunk["score_type"] = "cosine"
                elif mode == "bm25":
                    chunk["score_type"] = "bm25"
                else:
                    chunk["score_type"] = "rrf"
                results.append(chunk)

        return RetrievalResult(
            chunks=results,
            reranked=reranked,
            candidate_count=candidate_count,
            final_count=len(results),
            timings=timings
        )


def build_retriever(strategy: str, cfg: dict = None) -> Retriever:
    """Convenience function to build a Retriever for a strategy."""
    if cfg is None:
        cfg = load_config()
    from src.indexing import load_index
    bundle = load_index(strategy, cfg)
    return Retriever(bundle, cfg)


if __name__ == "__main__":
    # Quick smoke test
    cfg = load_config()

    for strategy in ["fixed", "semantic"]:
        print(f"\n=== Testing {strategy} ===")
        try:
            retriever = build_retriever(strategy, cfg)
            query = "What is retrieval-augmented generation?"

            for mode in ["dense", "bm25", "hybrid"]:
                for use_rerank in [False, True]:
                    result = retriever.retrieve(query, mode=mode, use_rerank=use_rerank, top_k=3)
                    print(f"  {mode} rerank={use_rerank}: {result.final_count} results (from {result.candidate_count} candidates)")
                    if result.chunks:
                        print(f"    Top: {result.chunks[0]['title'][:60]}... (score: {result.chunks[0]['score']:.4f})")
                    if result.timings:
                        print(f"    Timings: {result.timings}")
        except Exception as e:
            print(f"  Error: {e}")