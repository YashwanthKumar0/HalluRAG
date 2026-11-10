#!/usr/bin/env python
"""
Build chunks from processed papers for both fixed and semantic strategies.
"""
import json
import statistics
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.chunking import fixed_chunks, semantic_chunks
from src.data_prep import filter_chunks
from src.utils import read_jsonl, write_jsonl


def load_papers(processed_dir: Path) -> List[dict]:
    """Load all paper JSON files from processed directory."""
    papers = []
    for json_file in processed_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            papers.append(json.load(f))
    return papers


def print_stats(name: str, chunks: List[dict]):
    """Print statistics about chunks."""
    n_tokens = [c["n_tokens"] for c in chunks]
    print(f"\n{name}:")
    print(f"  Count: {len(chunks)}")
    print(f"  Mean tokens: {statistics.mean(n_tokens):.1f}")
    print(f"  Min tokens: {min(n_tokens)}")
    print(f"  Max tokens: {max(n_tokens)}")


def main():
    config = load_config()
    processed_dir = Path(config["data"]["processed_dir"])
    chunks_dir = Path(config["data"]["chunks_dir"])
    chunks_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = config["chunking"]["tokenizer"]
    fixed_cfg = config["chunking"]["fixed"]
    semantic_cfg = config["chunking"]["semantic"]
    min_words = config["chunking"].get("min_words", 40)
    min_alpha_ratio = config["chunking"].get("min_alpha_ratio", 0.6)

    papers = load_papers(processed_dir)
    print(f"Loaded {len(papers)} papers from {processed_dir}")

    # Fixed chunking
    print("\nBuilding fixed chunks...")
    all_fixed = []
    for paper in papers:
        chunks = fixed_chunks(
            paper,
            chunk_tokens=fixed_cfg["chunk_tokens"],
            overlap_tokens=fixed_cfg["overlap_tokens"],
            tokenizer_name=tokenizer_name
        )
        # Filter low-quality chunks
        chunks, dropped = filter_chunks(chunks, min_words=min_words, min_alpha_ratio=min_alpha_ratio)
        if dropped:
            print(f"  Dropped {dropped} low-quality chunks from {paper['paper_id']}")
        all_fixed.extend(chunks)

    fixed_path = chunks_dir / "chunks_fixed.jsonl"
    write_jsonl(str(fixed_path), all_fixed)
    print_stats("Fixed chunks", all_fixed)

    # Semantic chunking
    print("\nBuilding semantic chunks...")
    all_semantic = []
    for paper in papers:
        chunks = semantic_chunks(
            paper,
            max_tokens=semantic_cfg["max_tokens"],
            min_tokens=semantic_cfg["min_tokens"],
            tokenizer_name=tokenizer_name
        )
        # Filter low-quality chunks
        chunks, dropped = filter_chunks(chunks, min_words=min_words, min_alpha_ratio=min_alpha_ratio)
        if dropped:
            print(f"  Dropped {dropped} low-quality chunks from {paper['paper_id']}")
        all_semantic.extend(chunks)

    semantic_path = chunks_dir / "chunks_semantic.jsonl"
    write_jsonl(str(semantic_path), all_semantic)
    print_stats("Semantic chunks", all_semantic)

    print(f"\nDone. Chunks saved to {chunks_dir}")


if __name__ == "__main__":
    main()