#!/usr/bin/env python
"""
Export claims and evidence for human labeling.
Takes 20 answers from the best config (seed 42, mix of flags),
exports every claim with its evidence to data/gold/human_labels_todo.csv
with an empty human_label column.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.evaluation import run_evaluation
from src.retrieval import build_retriever
from src.generation import generate
from src.faithfulness import NLIScorer, verify_answer


def load_gold_set(gold_path: str) -> List[Dict]:
    """Load gold QA set."""
    data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_best_config(results_csv: str = "results/summary.csv") -> str:
    """Determine best config by faithfulness (fallback to recall@5)."""
    try:
        df = pd.read_csv(results_csv)
        # Prefer configs with faithfulness data
        if "faithfulness" in df.columns:
            df_f = df.dropna(subset=["faithfulness"])
            if not df_f.empty:
                best = df_f.loc[df_f["faithfulness"].idxmax(), "config"]
                return best
        # Fallback to recall@5
        if "recall@5" in df.columns:
            best = df.loc[df["recall@5"].idxmax(), "config"]
            return best
    except Exception:
        pass
    # Default
    return "semantic_dense_rerank0"


def export_for_human_labels(
    gold_path: str,
    output_path: str,
    config_name: str,
    cfg: Dict,
    n_samples: int = 20,
    seed: int = 42
):
    """Export claims and evidence for human labeling."""
    random.seed(seed)

    # Load gold set
    gold_items = load_gold_set(gold_path)

    # Build retriever
    parts = config_name.split("_")
    chunking = parts[0]
    retrieval = parts[1]
    rerank = parts[2] == "rerank1"

    retriever = build_retriever(chunking, cfg)

    # Initialize NLI scorer
    nli_model = cfg.get("faithfulness", {}).get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    nli_scorer = NLIScorer(nli_model)

    # Process all gold items to get claims and evidence
    all_claims = []

    for gold in gold_items:
        query = gold["question"]

        # Retrieve
        results = retriever.retrieve(
            query=query,
            mode=retrieval,
            use_rerank=rerank,
            top_k=cfg.get("retrieval", {}).get("candidate_k", 20)
        )

        # Generate answer
        top_k_chunks = results[:cfg.get("retrieval", {}).get("top_k", 5)]
        answer_data = generate(query, top_k_chunks, cfg)

        # Verify
        faithfulness_cfg = cfg.get("faithfulness", {})
        verify_result = verify_answer(
            answer_data.get("answer", ""),
            answer_data.get("citations", {}),
            results,
            nli_scorer,
            faithfulness_cfg
        )

        # Extract claims with evidence
        for claim_verdict in verify_result.get("claims", []):
            all_claims.append({
                "qid": gold.get("chunk_id", ""),
                "question": gold["question"],
                "gold_answer": gold.get("answer", ""),
                "generated_answer": answer_data.get("answer", ""),
                "claim": claim_verdict["claim"],
                "nli_label": claim_verdict["label"],  # supported/unverified/unsupported
                "nli_entail": claim_verdict["entail"],
                "nli_neutral": claim_verdict["neutral"],
                "nli_contradict": claim_verdict["contradict"],
                "evidence_chunk_id": claim_verdict["best_chunk_id"],
                "evidence_text": claim_verdict["best_evidence"][:500] if claim_verdict["best_evidence"] else "",
                "overall_flag": verify_result.get("overall_flag", ""),
                "human_label": ""  # Empty for human annotation
            })

    if not all_claims:
        print("No claims extracted!")
        return

    # Sample 20 claims ensuring mix of flags
    # Group by NLI label
    by_label = {}
    for c in all_claims:
        label = c["nli_label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(c)

    # Sample from each group
    sampled = []
    labels = list(by_label.keys())
    n_per_label = max(1, n_samples // len(labels))

    for label in labels:
        group = by_label[label]
        random.shuffle(group)
        sampled.extend(group[:n_per_label])

    # If not enough, add more randomly
    if len(sampled) < n_samples:
        remaining = [c for c in all_claims if c not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:n_samples - len(sampled)])

    sampled = sampled[:n_samples]

    # Write to CSV (hiding nli_label by putting it last)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "qid", "question", "gold_answer", "generated_answer",
        "claim", "evidence_chunk_id", "evidence_text",
        "nli_entail", "nli_neutral", "nli_contradict",
        "overall_flag",
        "human_label_a", "human_label_b", "human_label",  # For two annotators + final
        "nli_label"  # Hidden at the end
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow(row)

    print(f"Exported {len(sampled)} claims to {output_path}")
    print(f"Label distribution: {pd.Series([c['nli_label'] for c in sampled]).value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Export claims for human labeling")
    parser.add_argument("--gold", default="data/gold/gold_qa.jsonl", help="Gold QA set path")
    parser.add_argument("--output", default="data/gold/human_labels_todo.csv", help="Output CSV path")
    parser.add_argument("--config", help="Config name (default: best from summary.csv)")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of claims to export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_config()

    if args.config:
        config_name = args.config
    else:
        config_name = get_best_config()

    print(f"Using config: {config_name}")

    export_for_human_labels(
        args.gold,
        args.output,
        config_name,
        cfg,
        args.n_samples,
        args.seed
    )


if __name__ == "__main__":
    main()