"""
Tests for chunking module.
"""
import pytest
from src.chunking import fixed_chunking, semantic_chunking, Chunk
from src.utils import split_sentences


def test_fixed_chunking_basic():
    """Test basic fixed chunking."""
    text = "This is sentence one. This is sentence two. This is sentence three. " * 10
    chunks = fixed_chunking(text, "test_paper", "Test Paper", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_id.startswith("C")
        assert chunk.paper_id == "test_paper"
        assert chunk.strategy == "fixed"
        assert len(chunk.text) >= 50  # Minimum chunk size


def test_fixed_chunking_overlap():
    """Test that chunks have proper overlap."""
    text = "A " * 300  # ~600 chars
    chunks = fixed_chunking(text, "test", "Test", chunk_size=200, chunk_overlap=50)

    # Check overlap
    for i in range(len(chunks) - 1):
        # The end of chunk i should overlap with start of chunk i+1
        # Since we use character positions, just verify they're sequential
        assert chunks[i].end_char > chunks[i].start_char
        assert chunks[i+1].start_char < chunks[i].end_char


def test_fixed_chunking_short_text():
    """Test fixed chunking with text shorter than chunk_size."""
    text = "Short text."
    chunks = fixed_chunking(text, "test", "Test", chunk_size=512, chunk_overlap=50)

    # Should still create a chunk if text >= 50 chars
    # But our text is shorter, so might be 0 chunks
    # Actually the minimum is 50 chars, so this should be 0
    assert len(chunks) == 0


def test_semantic_chunking_basic():
    """Test basic semantic chunking."""
    text = "This is the first sentence. This is the second sentence. This is the third sentence. " * 5

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunks = semantic_chunking(
        text, "test_paper", "Test Paper", model,
        min_chunk_size=50, max_chunk_size=200, similarity_threshold=0.5
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.strategy == "semantic"


def test_split_sentences():
    """Test sentence splitting."""
    text = "This is sentence one. This is sentence two! And a third one?"
    sentences = split_sentences(text)

    assert len(sentences) == 3
    assert "sentence one" in sentences[0]
    assert "sentence two" in sentences[1]
    assert "third one" in sentences[2]


def test_chunk_id_generation():
    """Test deterministic chunk ID generation."""
    from src.utils import generate_chunk_id

    id1 = generate_chunk_id("same content")
    id2 = generate_chunk_id("same content")
    id3 = generate_chunk_id("different content")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("C")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])