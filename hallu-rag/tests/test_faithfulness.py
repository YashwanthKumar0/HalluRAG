"""
Tests for faithfulness module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
from src.faithfulness import FaithfulnessVerifier
from src.utils import split_sentences


def test_split_sentences():
    """Test sentence splitting utility."""
    text = "This is sentence one. This is sentence two! And a third one?"
    sentences = split_sentences(text)

    assert len(sentences) == 3
    assert "sentence one" in sentences[0]
    assert "sentence two" in sentences[1]
    assert "third one" in sentences[2]


def test_verifier_initialization():
    """Test FaithfulnessVerifier initialization with mocks."""
    with patch("src.faithfulness.AutoTokenizer") as mock_tokenizer, \
         patch("src.faithfulness.AutoModelForSequenceClassification") as mock_model:

        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        verifier = FaithfulnessVerifier(
            model_name="test-model",
            entailment_label=0,
            neutral_label=1,
            contradiction_label=2,
            confidence_threshold=0.5,
        )

        assert verifier.entailment_label == 0
        assert verifier.neutral_label == 1
        assert verifier.contradiction_label == 2
        assert verifier.confidence_threshold == 0.5


def test_verifier_verify_claim():
    """Test claim verification with mocked model."""
    with patch("src.faithfulness.AutoTokenizer") as mock_tokenizer, \
         patch("src.faithfulness.AutoModelForSequenceClassification") as mock_model_class:

        # Setup tokenizer mock
        mock_tok = Mock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tok.to.return_value = mock_tok.return_value
        mock_tokenizer.from_pretrained.return_value = mock_tok

        # Setup model mock
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Mock output with logits for entailment (label 0)
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0, 0.5, 0.1]])  # High entailment
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        verifier = FaithfulnessVerifier(model_name="test-model")
        result = verifier.verify_claim("The sky is blue", "The sky appears blue during the day")

        assert "claim" in result
        assert "label" in result
        assert "confidence" in result
        assert "probs" in result
        assert result["label"] == "supported"  # entailment_label=0
        assert result["confidence"] > 0.5


def test_verifier_verify_claim_contradiction():
    """Test claim verification for contradiction."""
    with patch("src.faithfulness.AutoTokenizer") as mock_tokenizer, \
         patch("src.faithfulness.AutoModelForSequenceClassification") as mock_model_class:

        mock_tok = Mock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tok.to.return_value = mock_tok.return_value
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Mock output with logits for contradiction (label 2)
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, 0.3, 2.0]])  # High contradiction
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        verifier = FaithfulnessVerifier(model_name="test-model")
        result = verifier.verify_claim("The sky is green", "The sky appears blue during the day")

        assert result["label"] == "unsupported"  # contradiction_label=2


def test_verifier_verify_answer():
    """Test answer verification with multiple claims."""
    with patch("src.faithfulness.AutoTokenizer") as mock_tokenizer, \
         patch("src.faithfulness.AutoModelForSequenceClassification") as mock_model_class:

        mock_tok = Mock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tok.to.return_value = mock_tok.return_value
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        # Return entailment for all
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0, 0.5, 0.1]])
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        verifier = FaithfulnessVerifier(model_name="test-model")

        answer = "The method improves accuracy. It uses a novel approach. Results show significant gains."
        chunks = [{"text": "The paper proposes a method that improves accuracy using a novel approach with significant gains."}]

        result = verifier.verify_answer(answer, chunks)

        assert "claims" in result
        assert "overall_faithfulness" in result
        assert "faithfulness_score" in result
        assert result["total_claims"] == 3
        assert result["supported_count"] == 3
        assert result["overall_faithfulness"] == "supported"


def test_format_verification():
    """Test verification formatting."""
    with patch("src.faithfulness.AutoTokenizer") as mock_tokenizer, \
         patch("src.faithfulness.AutoModelForSequenceClassification") as mock_model_class:

        mock_tok = Mock()
        mock_tok.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tok.to.return_value = mock_tok.return_value
        mock_tokenizer.from_pretrained.return_value = mock_tok

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_output = Mock()
        mock_output.logits = torch.tensor([[2.0, 0.5, 0.1]])
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        verifier = FaithfulnessVerifier(model_name="test-model")

        verification = {
            "overall_faithfulness": "supported",
            "faithfulness_score": 1.0,
            "supported_count": 2,
            "unverified_count": 0,
            "unsupported_count": 0,
            "total_claims": 2,
            "claims": [
                {"claim": "Test claim 1", "label": "supported", "confidence": 0.9, "emoji": "🟢"},
                {"claim": "Test claim 2", "label": "supported", "confidence": 0.8, "emoji": "🟢"},
            ],
        }

        formatted = verifier.format_verification(verification)
        assert "Overall Faithfulness" in formatted
        assert "Supported" in formatted
        assert "Claim 1" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])