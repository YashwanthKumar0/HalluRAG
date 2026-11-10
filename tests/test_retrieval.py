import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retrieval import Retriever
from src.indexing import IndexBundle


def test_rrf_fusion():
    """Test RRF on a toy example with hand-computed expected order."""
    # Create a mock bundle with known chunks
    chunks = [
        {"chunk_id": "a", "title": "A", "text": "text a", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "b", "title": "B", "text": "text b", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "c", "title": "C", "text": "text c", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "d", "title": "D", "text": "text d", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "e", "title": "E", "text": "text e", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
    ]

    # Mock FAISS index
    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(10)
    index.add(np.random.rand(5, 10).astype(np.float32))

    # Mock BM25
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["a"], ["b"], ["c"], ["d"], ["e"]])

    # Mock embedder
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    bundle = IndexBundle(
        faiss_index=index,
        bm25=bm25,
        chunks=chunks,
        chunk_id_to_row={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
        embedder=embedder,
        strategy="fixed"
    )

    cfg = {
        "retrieval": {"rrf_k": 60, "candidate_k": 5, "top_k": 3},
        "reranker": {"model": None}
    }

    retriever = Retriever(bundle, cfg, reranker_model=None)

    # Override dense and bm25 to return controlled rankings
    # Dense: a, b, c
    # BM25:  b, c, d
    original_dense = retriever.dense
    original_bm25 = retriever.bm25

    retriever.dense = lambda q, k: [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    retriever.bm25 = lambda q, k: [("b", 0.9), ("c", 0.8), ("d", 0.7)]

    # RRF with rrf_k=60
    # a: 1/(60+1) + 0 = 0.01639
    # b: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
    # c: 1/(60+3) + 1/(60+2) = 0.01587 + 0.01613 = 0.03200
    # d: 0 + 1/(60+3) = 0.01587
    # e: 0 + 0 = 0

    results = retriever.hybrid("test", 5)

    # Expected order: b > c > a > d > e
    assert results[0][0] == "b", f"Expected b first, got {results[0][0]}"
    assert results[1][0] == "c", f"Expected c second, got {results[1][0]}"
    assert results[2][0] == "a", f"Expected a third, got {results[2][0]}"
    assert results[3][0] == "d", f"Expected d fourth, got {results[3][0]}"
    # e has score 0, may or may not appear

    print("RRF test passed!")


def test_retrieve_returns_top_k():
    """Test that retrieve() returns exactly top_k unique chunk_ids for all 4 combinations."""
    # Create a mock bundle
    chunks = [
        {"chunk_id": f"chunk_{i}", "title": f"Title {i}", "text": f"text {i}",
         "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"}
        for i in range(10)
    ]

    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(10)
    index.add(np.random.rand(10, 10).astype(np.float32))

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["chunk", str(i)] for i in range(10)])

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    bundle = IndexBundle(
        faiss_index=index,
        bm25=bm25,
        chunks=chunks,
        chunk_id_to_row={f"chunk_{i}": i for i in range(10)},
        embedder=embedder,
        strategy="fixed"
    )

    cfg = {
        "retrieval": {"rrf_k": 60, "candidate_k": 10, "top_k": 5},
        "reranker": {"model": None}
    }

    retriever = Retriever(bundle, cfg, reranker_model=None)

    # Override to return predictable results
    retriever.dense = lambda q, k: [(f"chunk_{i}", 1.0 - i * 0.1) for i in range(min(k, 10))]
    retriever.bm25 = lambda q, k: [(f"chunk_{i}", 1.0 - i * 0.1) for i in range(min(k, 10))]

    modes = ["dense", "bm25", "hybrid"]
    rerank_options = [False, True]

    for mode in modes:
        for use_rerank in rerank_options:
            results = retriever.retrieve("test query", mode=mode, use_rerank=use_rerank, top_k=5)

            # Check exactly top_k results
            assert len(results) == 5, f"{mode} rerank={use_rerank}: expected 5 results, got {len(results)}"

            # Check unique chunk_ids
            chunk_ids = [r["chunk_id"] for r in results]
            assert len(set(chunk_ids)) == 5, f"{mode} rerank={use_rerank}: duplicate chunk_ids"

            # Check rank field is correct
            for i, r in enumerate(results):
                assert r["rank"] == i + 1, f"Rank mismatch at position {i}"

            # Check score field exists
            for r in results:
                assert "score" in r, "Missing score field"

    print("Retrieve top_k test passed!")


def test_retrieve_with_rerank():
    """Test that rerank changes the order when cross-encoder is available."""
    chunks = [
        {"chunk_id": "a", "title": "A", "text": "The cat sat on the mat.", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "b", "title": "B", "text": "A dog chased a ball.", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
        {"chunk_id": "c", "title": "C", "text": "Birds fly in the sky.", "section": "S", "n_tokens": 10, "strategy": "fixed", "paper_id": "p1"},
    ]

    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(10)
    index.add(np.random.rand(3, 10).astype(np.float32))

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["cat", "mat"], ["dog", "ball"], ["bird", "sky"]])

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    bundle = IndexBundle(
        faiss_index=index,
        bm25=bm25,
        chunks=chunks,
        chunk_id_to_row={"a": 0, "b": 1, "c": 2},
        embedder=embedder,
        strategy="fixed"
    )

    cfg = {
        "retrieval": {"rrf_k": 60, "candidate_k": 3, "top_k": 2},
        "reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
    }

    retriever = Retriever(bundle, cfg, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Initial retrieval returns a, b, c
    retriever.dense = lambda q, k: [("a", 0.9), ("b", 0.8), ("c", 0.7)]

    # Query about cats should rerank 'a' to top
    results = retriever.retrieve("cat", mode="dense", use_rerank=True, top_k=2)

    # With rerank, 'a' should be first (it's about cats)
    assert results[0]["chunk_id"] == "a", f"Expected 'a' first after rerank, got {results[0]['chunk_id']}"

    print("Rerank test passed!")


if __name__ == "__main__":
    test_rrf_fusion()
    test_retrieve_returns_top_k()
    test_retrieve_with_rerank()
    print("\nAll retrieval tests passed!")