"""
Chunking module for splitting documents into retrieval units.
Phase 2a: Fixed-size and semantic chunking strategies.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging, save_jsonl, load_jsonl, generate_chunk_id, clean_text, ensure_dir


logger = setup_logging()


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    strategy: str  # "fixed" or "semantic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fixed_chunking(
    text: str,
    paper_id: str,
    paper_title: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """
    Split text into fixed-size overlapping chunks.

    Args:
        text: Input text
        paper_id: Paper identifier
        paper_title: Paper title
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of Chunk objects
    """
    chunks = []
    text = clean_text(text)
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within the last 100 chars
            search_start = max(start + chunk_size - 100, start)
            sentence_end = text.rfind(". ", search_start, end)
            if sentence_end != -1:
                end = sentence_end + 1  # Include the period

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= 50:  # Minimum chunk size
            chunk_id = generate_chunk_id(f"{paper_id}_{chunk_index}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                text=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=chunk_index,
                strategy="fixed",
            ))
            chunk_index += 1

        start = end - chunk_overlap
        if start >= len(text):
            break

    return chunks


def semantic_chunking(
    text: str,
    paper_id: str,
    paper_title: str,
    model: SentenceTransformer,
    min_chunk_size: int = 100,
    max_chunk_size: int = 512,
    similarity_threshold: float = 0.75,
) -> List[Chunk]:
    """
    Split text into semantically coherent chunks using sentence embeddings.

    Args:
        text: Input text
        paper_id: Paper identifier
        paper_title: Paper title
        model: SentenceTransformer model for embeddings
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        similarity_threshold: Cosine similarity threshold for merging sentences

    Returns:
        List of Chunk objects
    """
    from .utils import split_sentences

    sentences = split_sentences(text)
    if not sentences:
        return []

    # Get embeddings for all sentences
    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_embedding = embeddings[0]
    current_start = 0
    chunk_index = 0

    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_emb = embeddings[i]

        # Compute similarity with current chunk
        similarity = np.dot(current_chunk_embedding, sent_emb)

        # Check if adding this sentence would exceed max size
        current_text = " ".join(current_chunk_sentences)
        potential_text = current_text + " " + sent

        if similarity >= similarity_threshold and len(potential_text) <= max_chunk_size:
            # Add to current chunk
            current_chunk_sentences.append(sent)
            # Update running average embedding
            current_chunk_embedding = (current_chunk_embedding * len(current_chunk_sentences) + sent_emb) / (len(current_chunk_sentences) + 1)
            current_chunk_embedding = current_chunk_embedding / np.linalg.norm(current_chunk_embedding)
        else:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= min_chunk_size:
                chunk_id = generate_chunk_id(f"{paper_id}_{chunk_index}")
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    chunk_index=chunk_index,
                    strategy="semantic",
                ))
                chunk_index += 1

            # Start new chunk
            current_chunk_sentences = [sent]
            current_chunk_embedding = sent_emb
            current_start += len(chunk_text) + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        if len(chunk_text) >= min_chunk_size:
            chunk_id = generate_chunk_id(f"{paper_id}_{chunk_index}")
            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title=paper_title,
                text=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                chunk_index=chunk_index,
                strategy="semantic",
            ))

    return chunks


def chunk_papers(
    processed_dir: Path,
    chunks_dir: Path,
    strategy: str = "fixed",
    **kwargs,
) -> List[Chunk]:
    """
    Chunk all papers in processed directory.

    Args:
        processed_dir: Directory with processed paper JSONs
        chunks_dir: Output directory for chunks
        strategy: Chunking strategy ("fixed" or "semantic")
        **kwargs: Additional arguments for chunking function

    Returns:
        List of all Chunk objects
    """
    ensure_dir(chunks_dir)

    # Load embedding model for semantic chunking
    model = None
    if strategy == "semantic":
        cfg = get_config()
        model_name = cfg.chunking.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        device = cfg.get_device()
        logger.info(f"Loading embedding model: {model_name} on {device}")
        model = SentenceTransformer(model_name, device=device)

    all_chunks = []
    paper_files = list(processed_dir.glob("*.json"))

    for paper_file in tqdm(paper_files, desc=f"Chunking papers ({strategy})"):
        with open(paper_file, "r") as f:
            paper = json.load(f)

        paper_id = paper["arxiv_id"]
        paper_title = paper["title"]
        text = paper.get("text", "")

        if not text:
            logger.warning(f"Paper {paper_id} has no text, skipping")
            continue

        if strategy == "fixed":
            chunk_size = kwargs.get("chunk_size", 512)
            chunk_overlap = kwargs.get("chunk_overlap", 50)
            chunks = fixed_chunking(text, paper_id, paper_title, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            min_chunk_size = kwargs.get("min_chunk_size", 100)
            max_chunk_size = kwargs.get("max_chunk_size", 512)
            similarity_threshold = kwargs.get("similarity_threshold", 0.75)
            chunks = semantic_chunking(
                text, paper_id, paper_title, model,
                min_chunk_size, max_chunk_size, similarity_threshold
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        all_chunks.extend(chunks)

    # Save chunks
    output_file = chunks_dir / f"chunks_{strategy}.jsonl"
    save_jsonl([c.to_dict() for c in all_chunks], output_file)
    logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")

    return all_chunks


def run_chunking(
    strategies: Optional[List[str]] = None,
    processed_dir: Optional[Path] = None,
    chunks_dir: Optional[Path] = None,
) -> Dict[str, List[Chunk]]:
    """
    Run chunking for specified strategies.

    Args:
        strategies: List of strategies to run (from config if None)
        processed_dir: Processed papers directory (from config if None)
        chunks_dir: Output chunks directory (from config if None)

    Returns:
        Dictionary mapping strategy to list of chunks
    """
    cfg = get_config()

    strategies = strategies or ["fixed", "semantic"]
    processed_dir = Path(processed_dir or cfg.data.get("processed_dir", "data/processed"))
    chunks_dir = Path(chunks_dir or cfg.data.get("chunks_dir", "data/chunks"))

    results = {}

    for strategy in strategies:
        logger.info(f"Running {strategy} chunking")

        if strategy == "fixed":
            chunk_kwargs = {
                "chunk_size": cfg.chunking.get("fixed", {}).get("chunk_size", 512),
                "chunk_overlap": cfg.chunking.get("fixed", {}).get("chunk_overlap", 50),
            }
        else:
            chunk_kwargs = {
                "min_chunk_size": cfg.chunking.get("semantic", {}).get("min_chunk_size", 100),
                "max_chunk_size": cfg.chunking.get("semantic", {}).get("max_chunk_size", 512),
                "similarity_threshold": cfg.chunking.get("semantic", {}).get("similarity_threshold", 0.75),
            }

        chunks = chunk_papers(processed_dir, chunks_dir, strategy, **chunk_kwargs)
        results[strategy] = chunks

    return results


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Chunking module loaded successfully")

    # Test fixed chunking
    test_text = "This is sentence one. This is sentence two. This is sentence three. " * 20
    chunks = fixed_chunking(test_text, "test_paper", "Test Paper", chunk_size=200, chunk_overlap=20)
    print(f"Fixed chunking: {len(chunks)} chunks")
    for c in chunks[:2]:
        print(f"  {c.chunk_id}: {c.text[:80]}...")

    print("Chunking smoke test passed!")