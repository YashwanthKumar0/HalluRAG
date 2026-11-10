import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import load_config
from src.utils import read_jsonl


@dataclass
class IndexBundle:
    """Container for all index components for a strategy."""
    faiss_index: faiss.IndexFlatIP
    bm25: BM25Okapi
    chunks: List[Dict]
    chunk_id_to_row: Dict[str, int]
    embedder: SentenceTransformer
    strategy: str
    bm25_corpus_tokens: List[List[str]] = None  # Pre-tokenized corpus for faster BM25
    model_name: str = ""
    search_timings: Dict[str, float] = None


# English stopwords for BM25
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "or", "but", "not", "have", "had",
    "his", "her", "she", "him", "they", "them", "their", "we", "our", "you",
    "your", "i", "me", "my", "mine", "myself", "yourself", "yourselves"
}


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text for BM25: lowercase, regex ``\\w+``, remove stopwords."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def format_doc(model_name: str, text: str) -> str:
    """Format document for embedding based on model type."""
    model_lower = model_name.lower()
    if "bge" in model_lower:
        # bge: no prefix for documents (only for queries)
        return text
    elif "e5" in model_lower:
        return f"passage: {text}"
    return text


def format_query(model_name: str, query: str) -> str:
    """Format query for embedding based on model type."""
    model_lower = model_name.lower()
    if "bge" in model_lower:
        return f"Represent this sentence for searching relevant passages: {query}"
    elif "e5" in model_lower:
        return f"query: {query}"
    return query


def load_chunks(chunks_path: Path) -> List[Dict]:
    """Load chunks from JSONL file."""
    return read_jsonl(str(chunks_path))


def build_faiss_index(chunks: List[Dict], embedder: SentenceTransformer, model_name: str, batch_size: int = 32, normalize: bool = True) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build FAISS index from chunks."""
    texts = []
    for chunk in chunks:
        doc_text = f"{chunk['title']}. {chunk['text']}"
        doc_text = format_doc(model_name, doc_text)
        texts.append(doc_text)

    # Generate embeddings
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    return index, embeddings


def build_bm25_index(chunks: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index from chunks. Returns (BM25, tokenized_corpus)."""
    corpus = []
    for chunk in chunks:
        text = f"{chunk['title']}. {chunk['text']}"
        corpus.append(tokenize_bm25(text))

    bm25 = BM25Okapi(corpus)
    return bm25, corpus


def save_index(strategy: str, index: faiss.IndexFlatIP, bm25: BM25Okapi,
               chunks: List[Dict], config: Dict,
               corpus_tokens: List[List[str]] = None):
    """Save all index components to disk."""
    index_dir = Path(config["data"]["index_dir"]) / strategy
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # Save chunk IDs mapping
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_id_to_row = {cid: i for i, cid in enumerate(chunk_ids)}
    with open(index_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_id_to_row, f)

    # Save chunks copy
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Save BM25
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    if corpus_tokens is not None:
        with open(index_dir / "bm25_corpus.pkl", "wb") as f:
            pickle.dump(corpus_tokens, f)

    print(f"Saved {strategy} index to {index_dir}")


def load_index(strategy: str, config: Dict = None) -> IndexBundle:
    """Load all index components for a strategy."""
    if config is None:
        config = load_config()

    index_dir = Path(config["data"]["index_dir"]) / strategy

    # Load FAISS
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

    # Load chunk IDs
    with open(index_dir / "chunk_ids.json", "r", encoding="utf-8") as f:
        chunk_id_to_row = json.load(f)

    # Load chunks
    chunks = read_jsonl(str(index_dir / "chunks.jsonl"))

    # Load BM25 and its tokenized corpus once at index-load time.
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    corpus_path = index_dir / "bm25_corpus.pkl"
    if corpus_path.exists():
        with open(corpus_path, "rb") as f:
            bm25_corpus_tokens = pickle.load(f)
    else:
        bm25_corpus_tokens = None

    # Load embedder
    model_name = config["embedding"]["model"]
    embedder = SentenceTransformer(model_name, device=config.get("device", "cpu"))

    return IndexBundle(
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        chunk_id_to_row=chunk_id_to_row,
        embedder=embedder,
        strategy=strategy,
        bm25_corpus_tokens=bm25_corpus_tokens,
        model_name=model_name,
    )


def build_all_indexes(config: Dict = None):
    """Build indexes for both fixed and semantic strategies."""
    if config is None:
        config = load_config()

    if "device" not in config:
        config["device"] = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if config["device"] == "cpu":
        import torch
        torch.set_num_threads(min(8, __import__("os").cpu_count() or 1))

    chunks_dir = Path(config["data"]["chunks_dir"])
    model_name = config["embedding"]["model"]
    batch_size = config["embedding"]["batch_size"]
    normalize = config["embedding"]["normalize"]

    embedder = SentenceTransformer(model_name, device=config["device"])

    for strategy in ["fixed", "semantic"]:
        print(f"\nBuilding indexes for {strategy}...")
        chunks_path = chunks_dir / f"chunks_{strategy}.jsonl"

        if not chunks_path.exists():
            print(f"  Chunks file not found: {chunks_path}")
            continue

        chunks = load_chunks(chunks_path)
        print(f"  Loaded {len(chunks)} chunks")

        # Build FAISS
        print("  Building FAISS index...")
        faiss_index, _ = build_faiss_index(chunks, embedder, model_name, batch_size, normalize)

        # Build BM25
        print("  Building BM25 index...")
        bm25, corpus_tokens = build_bm25_index(chunks)

        # Save
        save_index(strategy, faiss_index, bm25, chunks, config, corpus_tokens)

    print("\nAll indexes built!")


def dense_search(bundle: IndexBundle, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
    """Search FAISS index with a query."""
    model_name = bundle.model_name or load_config()["embedding"]["model"]

    formatted_query = format_query(model_name, query)
    query_started = time.perf_counter()
    query_emb = bundle.embedder.encode(
        [formatted_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)
    faiss_started = time.perf_counter()

    scores, indices = bundle.faiss_index.search(query_emb, k)
    bundle.search_timings = {
        "query_embedding": faiss_started - query_started,
        "faiss": time.perf_counter() - faiss_started,
    }

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((bundle.chunks[idx], float(score)))

    return results


def sparse_search(bundle: IndexBundle, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
    """Search BM25 index with a query. Uses numpy argpartition for faster top-k."""
    tokens = tokenize_bm25(query)
    scores = bundle.bm25.get_scores(tokens)

    # Use argpartition for O(n) top-k instead of full sort
    top_k = min(k, len(scores))
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((bundle.chunks[idx], float(scores[idx])))

    return results


if __name__ == "__main__":
    # Smoke test
    config = load_config()

    query = "What is retrieval-augmented generation?"

    for strategy in ["fixed", "semantic"]:
        print(f"\n=== {strategy.upper()} ===")
        try:
            bundle = load_index(strategy, config)

            # Dense search
            print("Dense (FAISS) top-3:")
            dense_results = dense_search(bundle, query, k=3)
            for i, (chunk, score) in enumerate(dense_results):
                print(f"  {i+1}. [{score:.4f}] {chunk['title'][:80]}... (section: {chunk['section']})")

            # Sparse search
            print("Sparse (BM25) top-3:")
            sparse_results = sparse_search(bundle, query, k=3)
            for i, (chunk, score) in enumerate(sparse_results):
                print(f"  {i+1}. [{score:.4f}] {chunk['title'][:80]}... (section: {chunk['section']})")
        except Exception as e:
            print(f"  Error loading {strategy}: {e}")