"""
Indexing module for building FAISS and BM25 retrieval indexes.
Phase 2b: Build vector and sparse indexes for each chunking strategy.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging, load_jsonl, ensure_dir, get_file_hash


logger = setup_logging()


class DenseIndex:
    """FAISS-based dense vector index."""

    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize dense index.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.chunk_ids: List[str] = []

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)
        elif self.index_type == "ivf":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """
        Add embeddings to index.

        Args:
            embeddings: Normalized embeddings (n_chunks, dimension)
            chunk_ids: List of chunk IDs
        """
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index.

        Args:
            query_embedding: Query embedding (1, dimension) or (n_queries, dimension)
            k: Number of results

        Returns:
            Tuple of (scores, indices)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return self.index.search(query_embedding, k)

    def save(self, path: Path) -> None:
        """Save index to disk."""
        ensure_dir(path.parent)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".ids.pkl"), "wb") as f:
            pickle.dump(self.chunk_ids, f)

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        """Load index from disk."""
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".ids.pkl"), "rb") as f:
            chunk_ids = pickle.load(f)

        obj = cls(index.d, "flat")  # type ignored, we replace index
        obj.index = index
        obj.chunk_ids = chunk_ids
        return obj


class BM25Index:
    """BM25 sparse retrieval index."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.corpus: List[List[str]] = []

    def add(self, texts: List[str], chunk_ids: List[str]) -> None:
        """
        Add documents to BM25 index.

        Args:
            texts: List of document texts
            chunk_ids: List of chunk IDs
        """
        # Tokenize
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.corpus = tokenized_corpus
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Lowercase, split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search BM25 index.

        Args:
            query: Query string
            k: Number of results

        Returns:
            Tuple of (scores, indices)
        """
        if self.bm25 is None:
            raise ValueError("Index not built")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]

        return top_scores, top_indices

    def save(self, path: Path) -> None:
        """Save BM25 index to disk."""
        ensure_dir(path.parent)
        data = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
        }
        with open(path.with_suffix(".bm25.pkl"), "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load BM25 index from disk."""
        with open(path.with_suffix(".bm25.pkl"), "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj.corpus = data["corpus"]
        obj.chunk_ids = data["chunk_ids"]
        obj.bm25 = BM25Okapi(obj.corpus)
        return obj


def build_dense_index(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    index_dir: Path,
    strategy: str,
    batch_size: int = 32,
) -> DenseIndex:
    """
    Build dense vector index for chunks.

    Args:
        chunks: List of chunk dictionaries
        model: SentenceTransformer model
        index_dir: Output directory
        strategy: Chunking strategy name
        batch_size: Batch size for encoding

    Returns:
        Built DenseIndex
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    logger.info(f"Encoding {len(texts)} chunks for dense index...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dimension = embeddings.shape[1]
    index = DenseIndex(dimension, "flat")
    index.add(embeddings, chunk_ids)

    # Save
    index_path = index_dir / f"dense_{strategy}"
    index.save(index_path)
    logger.info(f"Saved dense index to {index_path}")

    return index


def build_bm25_index(
    chunks: List[Dict[str, Any]],
    index_dir: Path,
    strategy: str,
) -> BM25Index:
    """
    Build BM25 index for chunks.

    Args:
        chunks: List of chunk dictionaries
        index_dir: Output directory
        strategy: Chunking strategy name

    Returns:
        Built BM25Index
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    logger.info(f"Building BM25 index for {len(texts)} chunks...")
    index = BM25Index()
    index.add(texts, chunk_ids)

    # Save
    index_path = index_dir / f"bm25_{strategy}"
    index.save(index_path)
    logger.info(f"Saved BM25 index to {index_path}")

    return index


def load_chunks(chunks_dir: Path, strategy: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks_path = chunks_dir / f"chunks_{strategy}.jsonl"
    return load_jsonl(chunks_path)


def build_all_indexes(
    strategies: Optional[List[str]] = None,
    chunks_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build all indexes for specified strategies.

    Args:
        strategies: List of chunking strategies (from config if None)
        chunks_dir: Chunks directory (from config if None)
        index_dir: Index output directory (from config if None)

    Returns:
        Dictionary mapping strategy to indexes
    """
    cfg = get_config()

    strategies = strategies or ["fixed", "semantic"]
    chunks_dir = Path(chunks_dir or cfg.data.get("chunks_dir", "data/chunks"))
    index_dir = Path(index_dir or cfg.data.get("index_dir", "data/index"))

    ensure_dir(index_dir)

    # Load embedding model
    model_name = cfg.retrieval.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    device = cfg.get_device()
    logger.info(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    results = {}

    for strategy in strategies:
        logger.info(f"Building indexes for {strategy} chunking")

        # Load chunks
        chunks = load_chunks(chunks_dir, strategy)
        if not chunks:
            logger.warning(f"No chunks found for {strategy}, skipping")
            continue

        # Build dense index
        dense_index = build_dense_index(chunks, model, index_dir, strategy)

        # Build BM25 index
        bm25_index = build_bm25_index(chunks, index_dir, strategy)

        results[strategy] = {
            "dense": dense_index,
            "bm25": bm25_index,
            "num_chunks": len(chunks),
        }

    return results


def load_indexes(
    strategies: Optional[List[str]] = None,
    index_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all indexes for specified strategies.

    Args:
        strategies: List of chunking strategies (from config if None)
        index_dir: Index directory (from config if None)

    Returns:
        Dictionary mapping strategy to indexes
    """
    cfg = get_config()

    strategies = strategies or ["fixed", "semantic"]
    index_dir = Path(index_dir or cfg.data.get("index_dir", "data/index"))

    results = {}

    for strategy in strategies:
        dense_path = index_dir / f"dense_{strategy}"
        bm25_path = index_dir / f"bm25_{strategy}"

        if not dense_path.with_suffix(".faiss").exists():
            logger.warning(f"Dense index not found for {strategy}")
            continue

        dense_index = DenseIndex.load(dense_path)
        bm25_index = BM25Index.load(bm25_path)

        results[strategy] = {
            "dense": dense_index,
            "bm25": bm25_index,
            "num_chunks": len(dense_index.chunk_ids),
        }

    return results


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Indexing module loaded successfully")
    print(f"Device: {cfg.get_device()}")