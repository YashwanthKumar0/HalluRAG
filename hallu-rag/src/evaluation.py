"""
Evaluation module for computing metrics on gold Q&A set.
Phase 6: Recall@k, MRR, faithfulness, answer relevance, human agreement.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import cohen_kappa_score, accuracy_score

from .config import get_config
from .pipeline import RAGPipeline, create_pipeline
from .utils import setup_logging, load_jsonl, save_json, save_jsonl


logger = setup_logging()


@dataclass
class EvaluationResult:
    """Results for a single experiment configuration."""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    per_question: List[Dict[str, Any]]


def compute_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_chunk_ids: List[str],
    k_values: List[int],
) -> Dict[int, float]:
    """
    Compute Recall@k for retrieved chunks.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        relevant_chunk_ids: List of relevant chunk IDs
        k_values: List of k values to compute recall at

    Returns:
        Dictionary mapping k to recall score
    """
    retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
    relevant_set = set(relevant_chunk_ids)

    recalls = {}
    for k in k_values:
        top_k_ids = set(retrieved_ids[:k])
        if len(relevant_set) == 0:
            recalls[k] = 1.0
        else:
            recalls[k] = len(top_k_ids & relevant_set) / len(relevant_set)

    return recalls


def compute_mrr(retrieved_chunks: List[Dict[str, Any]], relevant_chunk_ids: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries
        relevant_chunk_ids: List of relevant chunk IDs

    Returns:
        MRR score
    """
    retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
    relevant_set = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def compute_answer_relevance(
    answer: str,
    question: str,
    scorer: rouge_scorer.RougeScorer,
) -> Dict[str, float]:
    """
    Compute answer relevance using ROUGE-L between answer and question.

    Args:
        answer: Generated answer
        question: Original question
        scorer: ROUGE scorer

    Returns:
        Dictionary with ROUGE scores
    """
    scores = scorer.score(question, answer)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_faithfulness_metrics(verification: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute faithfulness metrics from verification results.

    Args:
        verification: Verification result dictionary

    Returns:
        Dictionary with faithfulness metrics
    """
    return {
        "faithfulness_score": verification.get("faithfulness_score", 0.0),
        "supported_ratio": verification.get("supported_count", 0) / max(verification.get("total_claims", 1), 1),
        "unsupported_ratio": verification.get("unsupported_count", 0) / max(verification.get("total_claims", 1), 1),
    }


def compute_human_agreement(
    predictions: List[str],
    human_labels: List[str],
) -> Dict[str, float]:
    """
    Compute agreement between system predictions and human labels.

    Args:
        predictions: System predicted labels
        human_labels: Human annotated labels

    Returns:
        Dictionary with agreement metrics
    """
    # Map labels to consistent format
    label_map = {
        "supported": "supported",
        "unsupported": "unsupported",
        "unverified": "unverified",
        "partially_supported": "unverified",
        "unknown": "unverified",
    }

    pred_mapped = [label_map.get(p, "unverified") for p in predictions]
    human_mapped = [label_map.get(h, "unverified") for h in human_labels]

    # Cohen's Kappa
    kappa = cohen_kappa_score(human_mapped, pred_mapped)

    # Accuracy
    acc = accuracy_score(human_mapped, pred_mapped)

    # Per-class agreement
    classes = ["supported", "unverified", "unsupported"]
    per_class = {}
    for cls in classes:
        pred_cls = [1 if p == cls else 0 for p in pred_mapped]
        human_cls = [1 if h == cls else 0 for h in human_mapped]
        if sum(human_cls) > 0:
            per_class[f"{cls}_precision"] = sum(p & h for p, h in zip(pred_cls, human_cls)) / max(sum(pred_cls), 1)
            per_class[f"{cls}_recall"] = sum(p & h for p, h in zip(pred_cls, human_cls)) / sum(human_cls)
        else:
            per_class[f"{cls}_precision"] = 0.0
            per_class[f"{cls}_recall"] = 0.0

    return {
        "cohen_kappa": kappa,
        "accuracy": acc,
        **per_class,
    }


def evaluate_pipeline(
    pipeline: RAGPipeline,
    gold_qa: List[Dict[str, Any]],
    k_values: List[int] = None,
    human_labels: Optional[List[Dict[str, Any]]] = None,
) -> EvaluationResult:
    """
    Evaluate a RAG pipeline on gold Q&A set.

    Args:
        pipeline: RAG pipeline to evaluate
        gold_qa: Gold Q&A list with question, answer, relevant_chunks
        k_values: k values for Recall@k (from config if None)
        human_labels: Optional human faithfulness labels

    Returns:
        EvaluationResult with metrics
    """
    cfg = get_config()
    k_values = k_values or cfg.evaluation.get("recall_k_values", [1, 3, 5, 10])

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    per_question_results = []
    all_recalls = {k: [] for k in k_values}
    all_mrr = []
    all_faithfulness = []
    all_relevance = {"rouge1_f": [], "rouge2_f": [], "rougeL_f": []}
    system_faithfulness = []
    human_faithfulness = []

    # Create mapping for human labels if provided
    human_label_map = {}
    if human_labels:
        for hl in human_labels:
            human_label_map[hl["question_id"]] = hl["faithfulness_label"]

    for qa in tqdm(gold_qa, desc="Evaluating"):
        question = qa["question"]
        question_id = qa.get("question_id", "")
        relevant_chunks = qa.get("relevant_chunks", [])

        # Run pipeline
        result = pipeline.run(question)

        # Compute retrieval metrics
        recalls = compute_recall_at_k(result.retrieved_chunks, relevant_chunks, k_values)
        for k, v in recalls.items():
            all_recalls[k].append(v)

        mrr = compute_mrr(result.retrieved_chunks, relevant_chunks)
        all_mrr.append(mrr)

        # Compute faithfulness metrics
        faith_metrics = compute_faithfulness_metrics(result.verification)
        all_faithfulness.append(faith_metrics["faithfulness_score"])

        # Compute answer relevance
        rel_metrics = compute_answer_relevance(result.answer, question, rouge)
        for k, v in rel_metrics.items():
            all_relevance[k].append(v)

        # Collect for human agreement
        system_faithfulness.append(result.verification["overall_faithfulness"])
        if question_id in human_label_map:
            human_faithfulness.append(human_label_map[question_id])

        # Store per-question result
        per_question_results.append({
            "question_id": question_id,
            "question": question,
            "answer": result.answer,
            "retrieval": {
                "recall_at_k": recalls,
                "mrr": mrr,
            },
            "faithfulness": faith_metrics,
            "relevance": rel_metrics,
            "verification": result.verification,
        })

    # Aggregate metrics
    metrics = {}

    # Retrieval metrics
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean(all_recalls[k])) if all_recalls[k] else 0.0
    metrics["mrr"] = float(np.mean(all_mrr)) if all_mrr else 0.0

    # Faithfulness metrics
    metrics["faithfulness_mean"] = float(np.mean(all_faithfulness)) if all_faithfulness else 0.0

    # Relevance metrics
    for k, v in all_relevance.items():
        metrics[f"relevance_{k}"] = float(np.mean(v)) if v else 0.0

    # Human agreement
    if human_faithfulness and len(human_faithfulness) == len(system_faithfulness):
        agreement = compute_human_agreement(system_faithfulness, human_faithfulness)
        metrics.update(agreement)

    return EvaluationResult(
        config=pipeline.config,
        metrics=metrics,
        per_question=per_question_results,
    )


def run_experiment(
    strategy: str,
    retrieval_type: str,
    use_rerank: bool,
    gold_qa_path: Path,
    human_labels_path: Optional[Path] = None,
    indexes: Optional[Dict[str, Any]] = None,
    chunks_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> EvaluationResult:
    """
    Run a single experiment configuration.

    Args:
        strategy: Chunking strategy
        retrieval_type: Retrieval type
        use_rerank: Whether to use reranking
        gold_qa_path: Path to gold Q&A file
        human_labels_path: Path to human labels file
        indexes: Pre-loaded indexes
        chunks_dir: Chunks directory
        index_dir: Index directory

    Returns:
        EvaluationResult
    """
    logger.info(f"Running experiment: {strategy}/{retrieval_type}/rerank={use_rerank}")

    # Load gold Q&A
    gold_qa = load_jsonl(gold_qa_path)

    # Load human labels if available
    human_labels = None
    if human_labels_path and human_labels_path.exists():
        human_labels = load_jsonl(human_labels_path)

    # Create pipeline
    pipeline = create_pipeline(
        strategy=strategy,
        retrieval_type=retrieval_type,
        use_rerank=use_rerank,
        indexes=indexes,
        chunks_dir=chunks_dir,
        index_dir=index_dir,
    )

    # Evaluate
    result = evaluate_pipeline(pipeline, gold_qa, human_labels=human_labels)

    return result


def run_all_experiments(
    gold_qa_path: Optional[Path] = None,
    human_labels_path: Optional[Path] = None,
    indexes: Optional[Dict[str, Any]] = None,
    chunks_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> List[EvaluationResult]:
    """
    Run all experiment configurations.

    Args:
        gold_qa_path: Gold Q&A path
        human_labels_path: Human labels path
        indexes: Pre-loaded indexes
        chunks_dir: Chunks directory
        index_dir: Index directory

    Returns:
        List of EvaluationResults
    """
    cfg = get_config()

    gold_qa_path = Path(gold_qa_path or cfg.evaluation.get("gold_qa_path", "data/gold/gold_qa.jsonl"))
    human_labels_path = Path(human_labels_path or cfg.evaluation.get("human_labels_path", "data/gold/human_labels.jsonl"))

    strategies = cfg.evaluation.get("experiments", {}).get("chunking_strategies", ["fixed", "semantic"])
    retrieval_types = cfg.evaluation.get("experiments", {}).get("retrieval_types", ["dense", "hybrid"])
    rerank_options = cfg.evaluation.get("experiments", {}).get("rerank_options", [False, True])

    results = []

    for strategy in strategies:
        for retrieval_type in retrieval_types:
            for use_rerank in rerank_options:
                result = run_experiment(
                    strategy=strategy,
                    retrieval_type=retrieval_type,
                    use_rerank=use_rerank,
                    gold_qa_path=gold_qa_path,
                    human_labels_path=human_labels_path,
                    indexes=indexes,
                    chunks_dir=chunks_dir,
                    index_dir=index_dir,
                )
                results.append(result)

    return results


def save_results(results: List[EvaluationResult], output_dir: Path) -> None:
    """
    Save experiment results.

    Args:
        results: List of EvaluationResults
        output_dir: Output directory
    """
    from .utils import ensure_dir
    ensure_dir(output_dir)

    # Save detailed results
    for i, result in enumerate(results):
        config_str = f"{result.config['strategy']}_{result.config['retrieval_type']}_rerank{result.config['use_rerank']}"
        save_json(result.per_question, output_dir / f"details_{config_str}.json")

    # Save summary table
    summary_rows = []
    for result in results:
        row = result.config.copy()
        row.update(result.metrics)
        summary_rows.append(row)

    save_json(summary_rows, output_dir / "summary.json")

    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)

    # Save markdown table
    with open(output_dir / "summary.md", "w") as f:
        f.write(df.to_markdown(index=False))

    logger.info(f"Saved results to {output_dir}")


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Evaluation module loaded successfully")
    print("Metrics: recall@k, mrr, faithfulness, answer_relevance, human_agreement")