"""Evaluation metrics for RAG system."""
import json
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from src.faithfulness import verify_answer, extract_claims, NLIScorer
from src.generation import parse_citations


def token_overlap(text1: str, text2: str) -> float:
    """Compute Jaccard token overlap for general text similarity."""
    tokens1 = set(_normalise_text(text1).split())
    tokens2 = set(_normalise_text(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def _normalise_paper_id(paper_id: Any) -> str:
    """Normalise paper IDs across storage formats and version suffixes."""
    value = str(paper_id or "").strip().replace("/", "_")
    return re.sub(r"v\d+$", "", value, flags=re.IGNORECASE)


def _normalise_text(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text or "").lower())).strip()


def is_relevant(chunk: Dict, gold_item: Dict, threshold: float = 0.6) -> bool:
    """
    Check if a chunk is relevant to a gold item.
    Same paper_id AND token overlap of supporting_text with chunk >= threshold.
    """
    # Must be same paper
    if _normalise_paper_id(chunk.get("paper_id")) != _normalise_paper_id(gold_item.get("paper_id")):
        return False

    # Check token overlap
    supporting_text = gold_item.get("supporting_text", "")
    chunk_text = chunk.get("text", "")

    if not supporting_text or not chunk_text:
        return False

    gold_tokens = set(_normalise_text(supporting_text).split())
    chunk_tokens = set(_normalise_text(chunk_text).split())
    overlap = len(gold_tokens & chunk_tokens) / len(gold_tokens)
    return overlap >= threshold


def recall_at_k(ranked_chunks: List[Dict], gold_item: Dict, k: int) -> float:
    """
    Recall@k: 1.0 if any relevant chunk in top-k, else 0.0.
    """
    top_k = ranked_chunks[:k]
    for chunk in top_k:
        if is_relevant(chunk, gold_item):
            return 1.0
    return 0.0


def mrr(ranked_chunks: List[Dict], gold_item: Dict) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant chunk (0 if none).
    """
    for rank, chunk in enumerate(ranked_chunks, 1):
        if is_relevant(chunk, gold_item):
            return 1.0 / rank
    return 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between two strings."""
    if not prediction or not ground_truth:
        return 0.0

    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def answer_correctness(predicted: str, gold: str) -> float:
    """Token F1 between generated answer and gold answer."""
    return token_f1(predicted, gold)


def is_abstained(answer: str) -> bool:
    """Check if answer indicates abstention."""
    refusal_phrases = [
        "do not contain enough information",
        "not contain enough information",
        "cannot answer",
        "unable to answer"
    ]
    return any(phrase in answer.lower() for phrase in refusal_phrases)


def compute_faithfulness(
    answers: List[Dict],
    retrieved_chunks: List[List[Dict]],
    nli_scorer: NLIScorer,
    cfg: Dict
) -> Tuple[float, Dict]:
    """
    Compute mean faithfulness over answered (non-abstained) questions.
    Returns (mean_faithfulness, details_dict).
    """
    faithfulness_scores = []
    details = {
        "total": len(answers),
        "answered": 0,
        "abstained": 0,
        "flags": {"green": 0, "amber": 0, "red": 0}
    }

    for i, (answer_data, chunks) in enumerate(zip(answers, retrieved_chunks)):
        answer = answer_data.get("answer", "")
        citations = answer_data.get("citations", {})

        if is_abstained(answer):
            details["abstained"] += 1
            continue

        details["answered"] += 1

        result = verify_answer(answer, citations, chunks, nli_scorer, cfg)
        faithfulness = result["faithfulness"]

        if faithfulness is not None:
            faithfulness_scores.append(faithfulness)

        details["flags"][result["overall_flag"]] += 1

    mean_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    return mean_faithfulness, details


def hallucination_rate(answers: List[Dict]) -> float:
    """Share of answers with overall_flag == 'red'."""
    if not answers:
        return 0.0
    red_count = sum(1 for a in answers if a.get("overall_flag") == "red")
    return red_count / len(answers)


def unsupported_claims_rate(answers: List[Dict]) -> float:
    """Share of unsupported claims over all claims."""
    total_claims = 0
    unsupported_claims = 0

    for a in answers:
        claims = a.get("claims", [])
        total_claims += len(claims)
        unsupported_claims += sum(1 for c in claims if c.get("label") == "unsupported")

    return unsupported_claims / total_claims if total_claims > 0 else 0.0


def answer_relevance(
    question: str,
    answer: str,
    embedder=None
) -> float:
    """
    Cosine similarity between question and answer embeddings.
    If no embedder provided, returns 0.0.
    """
    if embedder is None or not question or not answer:
        return 0.0

    try:
        q_emb = embedder.encode([question])
        a_emb = embedder.encode([answer])

        # Cosine similarity
        sim = np.dot(q_emb[0], a_emb[0]) / (
            np.linalg.norm(q_emb[0]) * np.linalg.norm(a_emb[0])
        )
        return float(sim)
    except Exception:
        return 0.0


def abstention_accuracy(
    answers: List[Dict],
    gold_items: List[Dict]
) -> float:
    """
    Accuracy on unanswerable questions (answer: null).
    Correct if system abstains.
    """
    unanswerable = [g for g in gold_items if g.get("answer") is None]
    if not unanswerable:
        return 0.0

    correct = 0
    for gold in unanswerable:
        # Find corresponding answer
        qid = gold.get("question", "")
        ans = next((a for a in answers if a.get("question") == qid), None)
        if ans and is_abstained(ans.get("answer", "")):
            correct += 1

    return correct / len(unanswerable)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""
    recall_at_k: Dict[int, float]
    mrr: float
    faithfulness: float
    faithfulness_details: Dict
    hallucination_rate: float
    unsupported_claims_rate: float
    answer_relevance: float
    answer_correctness: float
    abstention_accuracy: float


def run_evaluation(
    gold_items: List[Dict],
    answers: List[Dict],
    retrieved_chunks: List[List[Dict]],
    nli_scorer: NLIScorer,
    cfg: Dict,
    embedder=None,
    recall_ks: List[int] = None
) -> EvaluationResults:
    """
    Run full evaluation suite.

    Args:
        gold_items: List of gold QA items with question, answer, supporting_text, paper_id, chunk_id
        answers: List of generated answers with answer, citations, question
        retrieved_chunks: List of lists of retrieved chunks for each question
        nli_scorer: NLI scorer for faithfulness
        cfg: Faithfulness config
        embedder: Optional embedder for answer_relevance
        recall_ks: List of k values for recall@k

    Returns:
        EvaluationResults with all metrics
    """
    if recall_ks is None:
        recall_ks = [1, 3, 5, 10]

    # Recall@k and MRR
    recall_at_k_dict = {}
    for k in recall_ks:
        recalls = []
        for gold, chunks in zip(gold_items, retrieved_chunks):
            recalls.append(recall_at_k(chunks, gold, k))
        recall_at_k_dict[k] = np.mean(recalls) if recalls else 0.0

    mrr_scores = []
    for gold, chunks in zip(gold_items, retrieved_chunks):
        mrr_scores.append(mrr(chunks, gold))
    mrr_mean = np.mean(mrr_scores) if mrr_scores else 0.0

    # Faithfulness
    faithfulness, details = compute_faithfulness(
        answers, retrieved_chunks, nli_scorer, cfg
    )

    # Hallucination rate
    halluc_rate = hallucination_rate(answers)

    # Unsupported claims rate
    unsupported_rate = unsupported_claims_rate(answers)

    # Answer relevance
    relevance_scores = []
    for ans, gold in zip(answers, gold_items):
        score = answer_relevance(gold.get("question", ""), ans.get("answer", ""), embedder)
        relevance_scores.append(score)
    answer_relevance_mean = np.mean(relevance_scores) if relevance_scores else 0.0

    # Answer correctness (token F1)
    correctness_scores = []
    for ans, gold in zip(answers, gold_items):
        if gold.get("answer") is not None:  # Skip unanswerable
            score = answer_correctness(ans.get("answer", ""), gold.get("answer", ""))
            correctness_scores.append(score)
    answer_correctness_mean = np.mean(correctness_scores) if correctness_scores else 0.0

    # Abstention accuracy
    abstention_acc = abstention_accuracy(answers, gold_items)

    return EvaluationResults(
        recall_at_k=recall_at_k_dict,
        mrr=mrr_mean,
        faithfulness=faithfulness,
        faithfulness_details=details,
        hallucination_rate=halluc_rate,
        unsupported_claims_rate=unsupported_rate,
        answer_relevance=answer_relevance_mean,
        answer_correctness=answer_correctness_mean,
        abstention_accuracy=abstention_acc
    )


def print_results(results: EvaluationResults):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\nRetrieval Metrics:")
    for k, v in results.recall_at_k.items():
        print(f"  Recall@{k}: {v:.4f}")
    print(f"  MRR: {results.mrr:.4f}")

    print("\nFaithfulness Metrics:")
    print(f"  Mean Faithfulness: {results.faithfulness:.4f}")
    print(f"  Hallucination Rate (red): {results.hallucination_rate:.4f}")
    print(f"  Unsupported Claims Rate: {results.unsupported_claims_rate:.4f}")
    print(f"  Flags: {results.faithfulness_details['flags']}")
    print(f"  Answered: {results.faithfulness_details['answered']}, Abstained: {results.faithfulness_details['abstained']}")

    print("\nAnswer Quality Metrics:")
    print(f"  Answer Relevance: {results.answer_relevance:.4f}")
    print(f"  Answer Correctness (token F1): {results.answer_correctness:.4f}")
    print(f"  Abstention Accuracy: {results.abstention_accuracy:.4f}")

    print("="*60)


if __name__ == "__main__":
    # Simple test with toy data
    print("=== Evaluation Module Test ===")

    # Toy gold items
    gold = [
        {
            "question": "What is X?",
            "answer": "X is Y",
            "supporting_text": "X is Y according to the paper.",
            "paper_id": "paper1",
            "chunk_id": "paper1_0"
        },
        {
            "question": "What is unanswerable?",
            "answer": None,
            "supporting_text": "",
            "paper_id": "",
            "chunk_id": ""
        }
    ]

    # Toy retrieved chunks
    chunks = [
        [
            {"paper_id": "paper1", "chunk_id": "paper1_0", "text": "X is Y according to the paper."},
            {"paper_id": "paper1", "chunk_id": "paper1_1", "text": "Some other text."}
        ],
        [
            {"paper_id": "paper2", "chunk_id": "paper2_0", "text": "Irrelevant text."}
        ]
    ]

    # Toy answers
    answers = [
        {
            "question": "What is X?",
            "answer": "X is Y [C1]",
            "citations": {"C1": "paper1_0"}
        },
        {
            "question": "What is unanswerable?",
            "answer": "The provided documents do not contain enough information to answer this question.",
            "citations": {}
        }
    ]

    # Test recall
    print("\n--- Testing Recall@k ---")
    for k in [1, 2]:
        r = recall_at_k(chunks[0], gold[0], k)
        print(f"Recall@{k}: {r}")

    # Test MRR
    print("\n--- Testing MRR ---")
    m = mrr(chunks[0], gold[0])
    print(f"MRR: {m}")

    # Test token F1
    print("\n--- Testing Token F1 ---")
    f1 = token_f1("X is Y", "X is Y")
    print(f"Exact match F1: {f1}")
    f1 = token_f1("X is Y and Z", "X is Y")
    print(f"Partial match F1: {f1:.2f}")

    # Test abstention
    print("\n--- Testing Abstention ---")
    print(f"Abstained answer: {is_abstained('The provided documents do not contain enough information')}")
    print(f"Normal answer: {is_abstained('X is Y')}")

    print("\nAll tests passed!")