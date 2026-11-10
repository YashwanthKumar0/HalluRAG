"""
Tests for evaluation metrics.
"""
import pytest
import numpy as np
from src.evaluation import (
    compute_recall_at_k,
    compute_mrr,
    compute_answer_relevance,
    compute_faithfulness_metrics,
    compute_human_agreement,
)
from rouge_score import rouge_scorer


def test_compute_recall_at_k():
    """Test Recall@k computation."""
    retrieved = [
        {"chunk_id": "C1"},
        {"chunk_id": "C2"},
        {"chunk_id": "C3"},
        {"chunk_id": "C4"},
        {"chunk_id": "C5"},
    ]
    relevant = ["C1", "C3", "C5"]

    recalls = compute_recall_at_k(retrieved, relevant, [1, 3, 5])

    assert recalls[1] == 1/3  # Only C1 in top 1
    assert recalls[3] == 2/3  # C1, C3 in top 3
    assert recalls[5] == 1.0  # All 3 in top 5


def test_compute_recall_at_k_empty_relevant():
    """Test Recall@k with no relevant chunks."""
    retrieved = [{"chunk_id": "C1"}, {"chunk_id": "C2"}]
    relevant = []

    recalls = compute_recall_at_k(retrieved, relevant, [1, 2])
    assert recalls[1] == 1.0
    assert recalls[2] == 1.0


def test_compute_mrr():
    """Test MRR computation."""
    retrieved = [
        {"chunk_id": "C1"},
        {"chunk_id": "C2"},
        {"chunk_id": "C3"},
    ]
    relevant = ["C3", "C5"]

    mrr = compute_mrr(retrieved, relevant)
    assert mrr == 1/3  # First relevant at rank 3


def test_compute_mrr_first():
    """Test MRR when first result is relevant."""
    retrieved = [{"chunk_id": "C1"}, {"chunk_id": "C2"}]
    relevant = ["C1"]

    mrr = compute_mrr(retrieved, relevant)
    assert mrr == 1.0


def test_compute_mrr_none():
    """Test MRR when no relevant in retrieved."""
    retrieved = [{"chunk_id": "C1"}, {"chunk_id": "C2"}]
    relevant = ["C3", "C4"]

    mrr = compute_mrr(retrieved, relevant)
    assert mrr == 0.0


def test_compute_answer_relevance():
    """Test answer relevance with ROUGE."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    question = "What is the capital of France?"
    answer = "The capital of France is Paris."

    metrics = compute_answer_relevance(answer, question, scorer)

    assert "rouge1_f" in metrics
    assert "rouge2_f" in metrics
    assert "rougeL_f" in metrics
    assert 0 <= metrics["rouge1_f"] <= 1
    assert 0 <= metrics["rougeL_f"] <= 1


def test_compute_faithfulness_metrics():
    """Test faithfulness metrics computation."""
    verification = {
        "faithfulness_score": 0.8,
        "supported_count": 4,
        "unverified_count": 1,
        "unsupported_count": 0,
        "total_claims": 5,
    }

    metrics = compute_faithfulness_metrics(verification)

    assert metrics["faithfulness_score"] == 0.8
    assert metrics["supported_ratio"] == 0.8
    assert metrics["unsupported_ratio"] == 0.0


def test_compute_human_agreement():
    """Test human agreement computation."""
    predictions = ["supported", "supported", "unverified", "unsupported", "supported"]
    human_labels = ["supported", "unverified", "unverified", "unsupported", "supported"]

    agreement = compute_human_agreement(predictions, human_labels)

    assert "cohen_kappa" in agreement
    assert "accuracy" in agreement
    assert -1 <= agreement["cohen_kappa"] <= 1
    assert 0 <= agreement["accuracy"] <= 1


def test_compute_human_agreement_perfect():
    """Test human agreement with perfect match."""
    predictions = ["supported", "unverified", "unsupported"]
    human_labels = ["supported", "unverified", "unsupported"]

    agreement = compute_human_agreement(predictions, human_labels)

    assert agreement["accuracy"] == 1.0
    assert agreement["cohen_kappa"] == 1.0


def test_compute_human_agreement_mapping():
    """Test label mapping in human agreement."""
    # partially_supported should map to unverified
    predictions = ["partially_supported", "unknown"]
    human_labels = ["unverified", "unverified"]

    agreement = compute_human_agreement(predictions, human_labels)

    assert agreement["accuracy"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])