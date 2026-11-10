#!/usr/bin/env python
"""Print relevance diagnostics for the first five gold questions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.evaluation import is_relevant, _normalise_text
from src.retrieval import build_retriever


def load_gold(path: Path):
    import json
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    gold_items = load_gold(Path("data/gold/gold_qa.jsonl"))[:5]
    retriever = build_retriever("semantic", cfg)
    for index, gold in enumerate(gold_items, 1):
        result = retriever.retrieve(
            gold["question"], mode="hybrid", use_rerank=True, top_k=10
        )
        print(f"Question {index}: {gold['question']}")
        print(f"gold paper_id: {gold.get('paper_id', '')}")
        print(f"supporting_text: {gold.get('supporting_text', '')[:80]}")
        for rank, chunk in enumerate(result.chunks, 1):
            gold_tokens = set(_normalise_text(gold.get("supporting_text", "")).split())
            chunk_tokens = set(_normalise_text(chunk.get("text", "")).split())
            overlap = (
                len(gold_tokens & chunk_tokens) / len(gold_tokens)
                if gold_tokens else 0.0
            )
            print(
                f"  rank={rank} paper_id={chunk.get('paper_id', '')} "
                f"overlap={overlap:.3f} relevant={is_relevant(chunk, gold)}"
            )


if __name__ == "__main__":
    main()
