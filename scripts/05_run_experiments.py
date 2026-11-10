#!/usr/bin/env python
"""
Run evaluation experiments across all 8 configurations.

Configs: chunking ∈ {fixed, semantic} × retrieval ∈ {dense, hybrid} × rerank ∈ {off, on}

Outputs:
- results/per_question.csv
- results/summary.csv, results/summary.md
- results/ablation.md
- results/failure_cases.md
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.evaluation import (
    run_evaluation, print_results, EvaluationResults,
    is_abstained, recall_at_k, mrr
)
from src.retrieval import build_retriever
from src.generation import generate, build_prompt
from src.faithfulness import NLIScorer


CACHE_DIR = Path("results/cache")
PER_QUESTION_CSV = Path("results/per_question.csv")
SUMMARY_CSV = Path("results/summary.csv")
SUMMARY_MD = Path("results/summary.md")
ABLATION_MD = Path("results/ablation.md")
FAILURE_MD = Path("results/failure_cases.md")


def get_cache_path(config_name: str, qid: str) -> Path:
    """Get cache file path for a config-question pair."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{config_name}_{qid}.json"


def load_gold_set(gold_path: str) -> List[Dict]:
    """Load gold QA set."""
    data = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_config_name(chunking: str, retrieval: str, rerank: bool) -> str:
    """Generate config name string."""
    return f"{chunking}_{retrieval}_rerank{int(rerank)}"


def run_single_config(
    gold_items: List[Dict],
    chunking: str,
    retrieval: str,
    rerank: bool,
    cfg: Dict,
    nli_scorer: NLIScorer,
    embedder=None,
    retrieval_only: bool = False,
    use_cache: bool = True
) -> List[Dict]:
    """
    Run evaluation for a single configuration.
    Returns list of per-question result dicts.
    """
    config_name = get_config_name(chunking, retrieval, rerank)
    print(f"\n{'='*60}")
    print(f"Running config: {config_name}")
    print(f"{'='*60}")

    # Build retriever
    retriever = build_retriever(chunking, cfg)

    # Keep ten ranked chunks for retrieval evaluation. Generation still uses
    # the configured top_k prefix of this same ranked list.
    all_retrieved = []
    for gold in tqdm(gold_items, desc=f"Retrieving ({config_name})"):
        qid = gold.get("chunk_id", "")
        query = gold["question"]

        results = retriever.retrieve(
            query=query,
            mode=retrieval,
            use_rerank=rerank,
            top_k=10
        )
        all_retrieved.append(results)

    if retrieval_only:
        # Just compute retrieval metrics
        per_question = []
        for i, (gold, chunks) in enumerate(zip(gold_items, all_retrieved)):
            qid = gold.get("chunk_id", f"q{i}")

            # Compute retrieval metrics
            recalls = {}
            for k in [1, 3, 5, 10]:
                recalls[f"recall@{k}"] = recall_at_k(chunks, gold, k)
            recalls["mrr"] = mrr(chunks, gold)

            per_question.append({
                "config": config_name,
                "qid": qid,
                "question": gold["question"],
                **recalls,
                "faithfulness": None,
                "hallucination": None,
                "answer_relevance": None,
                "answer_correctness": None,
                "latency": None,
                "answer": None,
                "abstained": False,
                "claims": "[]"
                ,"status": "retrieval_only"
            })
        return per_question

    # Generation + verification
    per_question = []
    faithfulness_cfg = cfg.get("faithfulness", {})

    for i, (gold, chunks) in enumerate(tqdm(zip(gold_items, all_retrieved), desc=f"Generating ({config_name})")):
        qid = gold.get("chunk_id", f"q{i}")
        query = gold["question"]

        cache_path = get_cache_path(config_name, qid)

        # Check cache
        if use_cache and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                answer_data = cached.get("answer", {})
                retrieved_chunks = cached.get("retrieved_chunks", chunks)
                latency = cached.get("latency", 0)
        else:
            # Generate answer
            start_time = time.time()
            top_k_chunks = chunks[:cfg.get("retrieval", {}).get("top_k", 5)]

            # Use the same generation path as the application. It honors the
            # configured max_tokens and timeout and has no extractive fallback.
            answer_data = generate(query, top_k_chunks, cfg)

            latency = time.time() - start_time

            # Cache
            if use_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "answer": answer_data,
                        "retrieved_chunks": [c["chunk_id"] for c in chunks],
                        "latency": latency
                    }, f, ensure_ascii=False, indent=2)

        status = answer_data.get("status", "ok")
        failed = status == "llm_error"
        if failed:
            print(
                f"[{config_name}/{qid}] LLM failure: "
                f"{answer_data.get('error', 'unknown error')}"
            )
            faithfulness = hallucination = answer_relevance = answer_correctness = 0.0
            overall_flag = "error"
            claims = []
        # Verify faithfulness
        elif not is_abstained(answer_data.get("answer", "")):
            verify_result = run_evaluation(
                [gold], [answer_data], [chunks],
                nli_scorer, faithfulness_cfg, embedder
            )
            faithfulness = verify_result.faithfulness
            hallucination = verify_result.hallucination_rate
            answer_relevance = verify_result.answer_relevance
            answer_correctness = verify_result.answer_correctness
            overall_flag = (
                "red" if verify_result.hallucination_rate > 0 else "green"
            )
            claims = verify_result.faithfulness_details.get("claims", [])
        else:
            faithfulness = None
            hallucination = None
            answer_relevance = None
            answer_correctness = None
            overall_flag = "green"
            claims = []

        # Retrieval metrics
        recalls = {}
        for k in [1, 3, 5, 10]:
            recalls[f"recall@{k}"] = recall_at_k(chunks, gold, k)
        recalls["mrr"] = mrr(chunks, gold)

        per_question.append({
            "config": config_name,
            "qid": qid,
            "question": query,
            **recalls,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "answer_relevance": answer_relevance,
            "answer_correctness": answer_correctness,
            "overall_flag": overall_flag,
            "latency": latency,
            "answer": answer_data.get("answer", ""),
            "abstained": is_abstained(answer_data.get("answer", "")),
            "claims": json.dumps(claims)
            ,"status": status
        })

    return per_question


def write_per_question_csv(all_results: List[Dict]):
    """Write per-question results to CSV."""
    PER_QUESTION_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "config", "qid", "question",
        "recall@1", "recall@3", "recall@5", "recall@10", "mrr",
        "faithfulness", "hallucination", "answer_relevance",
        "answer_correctness", "overall_flag", "latency",
        "answer", "abstained", "claims", "status"
    ]

    with open(PER_QUESTION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)


def write_summary_csv(all_results: List[Dict]):
    """Write summary statistics per config to CSV."""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Group by config
    df = pd.DataFrame(all_results)

    # Handle None values for numeric columns
    numeric_cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr",
                    "faithfulness", "hallucination", "answer_relevance",
                    "answer_correctness", "latency"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Compute mean per config
    summary = df.groupby("config")[numeric_cols].mean().reset_index()
    summary["n_questions"] = df.groupby("config").size().values

    # Handle abstained column (may have None values)
    df["abstained"] = df["abstained"].fillna(False)
    summary["n_answered"] = df[~df["abstained"]].groupby("config").size().reindex(summary["config"], fill_value=0).values
    if "status" in df.columns:
        summary["n_failed"] = (
            df[df["status"] == "llm_error"].groupby("config").size()
            .reindex(summary["config"], fill_value=0).values
        )

    # Flag counts
    if "overall_flag" in df.columns:
        flag_counts = df.groupby(["config", "overall_flag"]).size().unstack(fill_value=0)
        for flag in ["green", "amber", "red"]:
            if flag in flag_counts.columns:
                summary[f"flag_{flag}"] = flag_counts[flag].reindex(summary["config"], fill_value=0).values

    summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")
    return summary


def write_summary_md(summary_df: pd.DataFrame):
    """Write summary as markdown table."""
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"Total configs: {len(summary_df)}\n\n")

        # Main metrics table
        cols = ["config", "n_questions", "n_answered", "n_failed",
                "recall@1", "recall@3", "recall@5", "recall@10", "mrr",
                "faithfulness", "hallucination", "answer_relevance",
                "answer_correctness", "latency"]

        available_cols = [c for c in cols if c in summary_df.columns]
        f.write(summary_df[available_cols].to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        # Flag table
        flag_cols = [c for c in summary_df.columns if c.startswith("flag_")]
        if flag_cols:
            f.write("## Flag Distribution\n\n")
            f.write(summary_df[["config"] + flag_cols].to_markdown(index=False))
            f.write("\n\n")


def write_ablation_md(all_results: List[Dict]):
    """Write ablation tables isolating each factor."""
    ABLATION_MD.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    numeric_cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr",
                    "faithfulness", "hallucination", "answer_relevance",
                    "answer_correctness", "latency"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    with open(ABLATION_MD, "w", encoding="utf-8") as f:
        f.write("# Ablation Study\n\n")
        f.write("Each table averages over the other two factors.\n\n")

        # 1. Chunking: fixed vs semantic
        f.write("## 1. Chunking Strategy (fixed vs semantic)\n\n")
        chunking_metrics = df.groupby("config")[numeric_cols].mean().reset_index()
        # Extract fixed vs semantic from config name
        chunking_metrics["chunking"] = chunking_metrics["config"].apply(
            lambda x: x.split("_")[0]
        )
        chunking_avg = chunking_metrics.groupby("chunking")[numeric_cols].mean().reset_index()
        f.write(chunking_avg.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        # 2. Retrieval: dense vs hybrid
        f.write("## 2. Retrieval Method (dense vs hybrid)\n\n")
        chunking_metrics["retrieval"] = chunking_metrics["config"].apply(
            lambda x: x.split("_")[1]
        )
        retrieval_avg = chunking_metrics.groupby("retrieval")[numeric_cols].mean().reset_index()
        f.write(retrieval_avg.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        # 3. Rerank: off vs on
        f.write("## 3. Reranking (off vs on)\n\n")
        chunking_metrics["rerank"] = chunking_metrics["config"].apply(
            lambda x: x.split("_")[2]
        )
        rerank_avg = chunking_metrics.groupby("rerank")[numeric_cols].mean().reset_index()
        f.write(rerank_avg.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")


def write_failure_cases_md(all_results: List[Dict], gold_items: List[Dict]):
    """Write failure cases for each of the 3 comparisons."""
    FAILURE_MD.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    numeric_cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr",
                    "faithfulness", "hallucination", "answer_relevance",
                    "answer_correctness", "latency"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    with open(FAILURE_MD, "w", encoding="utf-8") as f:
        f.write("# Failure Case Analysis\n\n")
        f.write("For each factor comparison, the 3 questions with largest metric difference.\n\n")

        # Map qid to gold item
        gold_by_qid = {g.get("chunk_id", f"q{i}"): g for i, g in enumerate(gold_items)}

        # 1. Chunking
        f.write("## 1. Chunking: fixed vs semantic\n\n")
        _write_factor_failures(f, df, gold_by_qid, "chunking",
                               lambda c: c.split("_")[0], "fixed", "semantic",
                               "faithfulness")

        # 2. Retrieval
        f.write("## 2. Retrieval: dense vs hybrid\n\n")
        _write_factor_failures(f, df, gold_by_qid, "retrieval",
                               lambda c: c.split("_")[1], "dense", "hybrid",
                               "faithfulness")

        # 3. Rerank
        f.write("## 3. Rerank: off vs on\n\n")
        _write_factor_failures(f, df, gold_by_qid, "rerank",
                               lambda c: c.split("_")[2], "rerank0", "rerank1",
                               "faithfulness")


def _write_factor_failures(
    f, df: pd.DataFrame, gold_by_qid: Dict, factor_name: str,
    extract_fn, val_a: str, val_b: str, metric: str
):
    """Helper to write failure cases for one factor."""
    # Get configs for each value
    configs_a = df[df["config"].apply(extract_fn) == val_a]["config"].unique()
    configs_b = df[df["config"].apply(extract_fn) == val_b]["config"].unique()

    # Average metric per qid for each setting
    df_a = df[df["config"].apply(extract_fn) == val_a].groupby("qid")[metric].mean()
    df_b = df[df["config"].apply(extract_fn) == val_b].groupby("qid")[metric].mean()

    # Find differences
    common_qids = set(df_a.index) & set(df_b.index)
    diffs = {qid: abs(df_a[qid] - df_b[qid]) for qid in common_qids
             if not np.isnan(df_a[qid]) and not np.isnan(df_b[qid])}

    # Top 3
    top_qids = sorted(diffs, key=diffs.get, reverse=True)[:3]

    for rank, qid in enumerate(top_qids, 1):
        gold = gold_by_qid.get(qid, {})
        f.write(f"### Question {rank}: {qid} (Δ{diffs[qid]:.4f})\n\n")
        f.write(f"**Question**: {gold.get('question', 'N/A')}\n\n")
        f.write(f"**Gold Answer**: {gold.get('answer', 'N/A')}\n\n")
        f.write(f"**Supporting Text**: {gold.get('supporting_text', 'N/A')[:500]}...\n\n")

        # Get both answers
        for val in [val_a, val_b]:
            configs = configs_a if val == val_a else configs_b
            row = df[(df["qid"] == qid) & (df["config"].isin(configs))]
            if not row.empty:
                r = row.iloc[0]
                f.write(f"**{factor_name}={val}**:\n")
                f.write(f"- Answer: {r['answer'][:300]}...\n")
                f.write(f"- Flag: {r['overall_flag']}\n")
                f.write(f"- Faithfulness: {r['faithfulness']}\n")
                f.write(f"- Latency: {r['latency']:.2f}s\n\n")

        # Top-3 chunks for both settings
        for val, configs in [(val_a, configs_a), (val_b, configs_b)]:
            f.write(f"**Top-3 chunks ({factor_name}={val})**:\n")
            # Would need to store retrieved chunk IDs in results for this
            f.write("- (chunk details not stored in this run)\n")

        f.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--gold", default="data/gold/gold_qa.jsonl", help="Gold QA set path")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--configs", nargs="+", help="Specific configs to run (e.g., fixed_dense_rerank0)")
    parser.add_argument("--ragas", action="store_true", help="Use RAGAS metrics if available")
    args = parser.parse_args()

    # Load config and gold set
    cfg = load_config()
    gold_items = load_gold_set(args.gold)
    print(f"Loaded {len(gold_items)} gold questions")

    # Retrieval-only runs do not need to load the NLI model.
    nli_scorer = None
    if not args.retrieval_only:
        nli_model = cfg.get("faithfulness", {}).get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        print(f"Loading NLI model: {nli_model}...")
        nli_scorer = NLIScorer(nli_model)

    # Initialize embedder for answer_relevance
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"))
    except ImportError:
        pass

    # Determine configs to run
    if args.configs:
        configs_to_run = []
        for c in args.configs:
            parts = c.split("_")
            configs_to_run.append({
                "chunking": parts[0],
                "retrieval": parts[1],
                "rerank": parts[2] == "rerank1"
            })
    else:
        configs_to_run = [
            {"chunking": "fixed", "retrieval": "dense", "rerank": False},
            {"chunking": "fixed", "retrieval": "dense", "rerank": True},
            {"chunking": "fixed", "retrieval": "hybrid", "rerank": False},
            {"chunking": "fixed", "retrieval": "hybrid", "rerank": True},
            {"chunking": "semantic", "retrieval": "dense", "rerank": False},
            {"chunking": "semantic", "retrieval": "dense", "rerank": True},
            {"chunking": "semantic", "retrieval": "hybrid", "rerank": False},
            {"chunking": "semantic", "retrieval": "hybrid", "rerank": True},
        ]

    # Run all configs
    all_results = []
    for config in configs_to_run:
        results = run_single_config(
            gold_items,
            config["chunking"],
            config["retrieval"],
            config["rerank"],
            cfg,
            nli_scorer,
            embedder=embedder,
            retrieval_only=args.retrieval_only,
            use_cache=not args.no_cache
        )
        all_results.extend(results)

    # Write outputs
    write_per_question_csv(all_results)
    summary_df = write_summary_csv(all_results)
    write_summary_md(summary_df)
    write_ablation_md(all_results)
    write_failure_cases_md(all_results, gold_items)

    print(f"\n{'='*60}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Per-question: {PER_QUESTION_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary MD: {SUMMARY_MD}")
    print(f"Ablation MD: {ABLATION_MD}")
    print(f"Failure cases: {FAILURE_MD}")


if __name__ == "__main__":
    main()