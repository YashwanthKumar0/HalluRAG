"""
Tests for retrieval module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.retrieval import Retriever, DenseIndex, BM25Index
from src.utils import generate_chunk_id


def test_dense_index_add_search():
    """Test DenseIndex add and search."""
    # Create mock embeddings
    embeddings = np.random.rand(10, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunk_ids = [generate_chunk_id(f"chunk_{i}") for i in range(10)]

    index = DenseIndex(384, "flat")
    index.add(embeddings, chunk_ids)

    assert len(index.chunk_ids) == 10

    # Search
    query = embeddings[0:1]
    scores, indices = index.search(query, 5)

    assert scores.shape == (1, 5)
    assert indices.shape == (1, 5)
    # First result should be the same chunk (highest score)
    assert indices[0, 0] == 0
    assert scores[0, 0] > 0.99  # Very high similarity


def test_bm25_index_add_search():
    """Test BM25Index add and search."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over the lazy fox",
        "The lazy dog sleeps while the fox runs",
    ]
    chunk_ids = [generate_chunk_id(f"chunk_{i}") for i in range(3)]

    index = BM25Index()
    index.add(texts, chunk_ids)

    assert len(index.chunk_ids) == 3

    # Search for "fox"
    scores, indices = index.search("fox", 2)

    assert len(scores) == 2
    assert len(indices) == 2
    # First result should contain "fox"
    assert "fox" in texts[indices[0]].lower()


def test_retriever_dense_retrieval():
    """Test Retriever dense retrieval with mocks."""
    with patch("src.retrieval.SentenceTransformer") as mock_st, \
         patch("src.retrieval.load_indexes") as mock_load:

        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        mock_dense = Mock()
        mock_dense.chunk_ids = [generate_chunk_id(f"c{i}") for i in range(5)]
        mock_dense.search.return_value = (np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]), np.array([[0, 1, 2, 3, 4]]))

        mock_bm25 = Mock()
        mock_bm25.chunk_ids = [generate_chunk_id(f"c{i}") for i in range(5)]

        mock_load.return_value = {
            "fixed": {"dense": mock_dense, "bm25": mock_bm25, "num_chunks": 5}
        }

        retriever = Retriever(strategy="fixed", retrieval_type="dense", use_rerank=False)

        # Mock chunk_texts
        retriever.chunk_texts = {cid: f"Text for {cid}" for cid in mock_dense.chunk_ids}

        results = retriever.retrieve("test query", top_k=3)

        assert len(results) == 3
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("citation_id" in r for r in results)


def test_retriever_hybrid_retrieval():
    """Test Retriever hybrid retrieval with mocks."""
    with patch("src.retrieval.SentenceTransformer") as mock_st, \
         patch("src.retrieval.load_indexes") as mock_load:

        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_st.return_value = mock_model

        mock_dense = Mock()
        mock_dense.chunk_ids = [generate_chunk_id(f"c{i}") for i in range(5)]
        mock_dense.search.return_value = (np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]), np.array([[0, 1, 2, 3, 4]]))

        mock_bm25 = Mock()
        mock_bm25.chunk_ids = [generate_chunk_id(f"c{i}") for i in range(5)]
        mock_bm25.search.return_value = (np.array([0.7, 0.6, 0.5, 0.4, 0.3]), np.array([0, 1, 2, 3, 4]))

        mock_load.return_value = {
            "fixed": {"dense": mock_dense, "bm25": mock_bm25, "num_chunks": 5}
        }

        retriever = Retriever(strategy="fixed", retrieval_type="hybrid", use_rerank=False)
        retriever.chunk_texts = {cid: f"Text for {cid}" for cid in mock_dense.chunk_ids}

        results = retriever.retrieve("test query", top_k=3)

        assert len(results) == 3
        assert all("citation_id" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])