"""
Faithfulness verification module using NLI (Natural Language Inference).
Phase 5: Verify each sentence-level claim against retrieved evidence.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging, split_sentences, clean_text


logger = setup_logging()


class FaithfulnessVerifier:
    """Verify faithfulness of generated answers using NLI."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        entailment_label: Optional[int] = None,
        neutral_label: Optional[int] = None,
        contradiction_label: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize faithfulness verifier.

        Args:
            model_name: NLI model name (from config if None)
            entailment_label: Label index for entailment (from config if None)
            neutral_label: Label index for neutral (from config if None)
            contradiction_label: Label index for contradiction (from config if None)
            confidence_threshold: Confidence threshold (from config if None)
        """
        cfg = get_config()
        faith_config = cfg.faithfulness

        self.model_name = model_name or faith_config.get("nli_model", "microsoft/deberta-v3-base-mnli")
        self.entailment_label = entailment_label or faith_config.get("entailment_label", 0)
        self.neutral_label = neutral_label or faith_config.get("neutral_label", 1)
        self.contradiction_label = contradiction_label or faith_config.get("contradiction_label", 2)
        self.confidence_threshold = confidence_threshold or faith_config.get("confidence_threshold", 0.5)

        self.device = cfg.get_device()
        logger.info(f"Loading NLI model: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.id2label = {
            self.entailment_label: "supported",
            self.neutral_label: "unverified",
            self.contradiction_label: "unsupported",
        }

        self.label2emoji = {
            "supported": "🟢",
            "unverified": "🟡",
            "unsupported": "🔴",
        }

    def verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Verify a single claim against evidence using NLI.

        Args:
            claim: Claim sentence to verify
            evidence: Evidence text

        Returns:
            Verification result dictionary
        """
        # Truncate evidence if too long (NLI models typically have 512 token limit)
        max_evidence_length = 400  # Leave room for claim + special tokens
        if len(evidence) > max_evidence_length:
            evidence = evidence[:max_evidence_length]

        # Prepare input: premise = evidence, hypothesis = claim
        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Get predicted label and confidence
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        label = self.id2label.get(pred_idx, "unknown")

        return {
            "claim": claim,
            "label": label,
            "confidence": confidence,
            "probs": {
                "supported": float(probs[self.entailment_label]) if self.entailment_label < len(probs) else 0.0,
                "unverified": float(probs[self.neutral_label]) if self.neutral_label < len(probs) else 0.0,
                "unsupported": float(probs[self.contradiction_label]) if self.contradiction_label < len(probs) else 0.0,
            },
            "emoji": self.label2emoji.get(label, "⚪"),
        }

    def verify_answer(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify all claims in an answer against retrieved evidence.

        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved chunks used for generation

        Returns:
            Verification results for all claims
        """
        # Combine all retrieved text as evidence
        evidence_text = " ".join([c.get("text", "") for c in retrieved_chunks])
        evidence_text = clean_text(evidence_text)

        # Split answer into sentences (claims)
        claims = split_sentences(answer)

        if not claims:
            return {
                "claims": [],
                "overall_faithfulness": "unknown",
                "supported_count": 0,
                "unverified_count": 0,
                "unsupported_count": 0,
            }

        # Verify each claim
        claim_results = []
        for claim in claims:
            claim = claim.strip()
            if len(claim) < 10:  # Skip very short fragments
                continue

            result = self.verify_claim(claim, evidence_text)
            claim_results.append(result)

        # Count labels
        supported_count = sum(1 for r in claim_results if r["label"] == "supported")
        unverified_count = sum(1 for r in claim_results if r["label"] == "unverified")
        unsupported_count = sum(1 for r in claim_results if r["label"] == "unsupported")

        # Determine overall faithfulness
        total = len(claim_results)
        if total == 0:
            overall = "unknown"
        elif unsupported_count > 0:
            overall = "unsupported"
        elif supported_count == total:
            overall = "supported"
        else:
            overall = "partially_supported"

        return {
            "claims": claim_results,
            "overall_faithfulness": overall,
            "supported_count": supported_count,
            "unverified_count": unverified_count,
            "unsupported_count": unsupported_count,
            "total_claims": total,
            "faithfulness_score": supported_count / total if total > 0 else 0.0,
        }

    def format_verification(self, verification: Dict[str, Any]) -> str:
        """
        Format verification results for display.

        Args:
            verification: Verification result dictionary

        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Overall Faithfulness: {verification['overall_faithfulness']}")
        lines.append(f"Score: {verification['faithfulness_score']:.2f}")
        lines.append(f"Claims: {verification['supported_count']}🟢 {verification['unverified_count']}🟡 {verification['unsupported_count']}🔴")
        lines.append("")

        for i, claim in enumerate(verification["claims"]):
            lines.append(f"Claim {i+1}: {claim['emoji']} {claim['label']} ({claim['confidence']:.2f})")
            lines.append(f"  Text: {claim['claim']}")

        return "\n".join(lines)


def create_verifier(
    model_name: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
) -> FaithfulnessVerifier:
    """Factory function to create a verifier."""
    return FaithfulnessVerifier(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
    )


if __name__ == "__main__":
    # Smoke test
    cfg = get_config()
    print("Faithfulness module loaded successfully")
    print(f"NLI Model: {cfg.faithfulness.get('nli_model')}")
    print(f"Device: {cfg.get_device()}")

    # Test sentence splitting
    from .utils import split_sentences
    test = "This is a test. This is another sentence! And a third one?"
    print(f"Sentences: {split_sentences(test)}")