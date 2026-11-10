import os
import re
import time
import nltk
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


_NLI_SCORER_CACHE: Dict[str, "NLIScorer"] = {}
torch.set_num_threads(os.cpu_count() or 1)


class Claim(tuple):
    """Tuple-compatible claim with text and citation tags."""

    def __new__(cls, text: str, citations: List[str]):
        return super().__new__(cls, (text, citations))

    @property
    def text(self) -> str:
        return self[0]

    @property
    def citations(self) -> List[str]:
        return self[1]

    def __contains__(self, value: object) -> bool:
        if isinstance(value, str):
            return value in self.text or value in self.citations
        return super().__contains__(value)

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


@dataclass
class NLIScorer:
    """NLI model for entailment/neutral/contradiction scoring."""
    model_name: str
    max_length: int = 512

    def __post_init__(self):
        print(f"Loading NLI model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

        # Read label order from config - never assume index order
        self.id2label = self.config.id2label
        self.entail_idx = self._find_label_index("entail")
        self.neutral_idx = self._find_label_index("neutral")
        self.contradict_idx = self._find_label_index("contradict")

    def _find_label_index(self, label_fragment: str) -> int:
        for index, label in self.id2label.items():
            if label_fragment in str(label).lower():
                return int(index)
        raise ValueError(f"NLI model has no label containing {label_fragment!r}: {self.id2label}")

    def score(self, premises: List[str], hypothesis: str) -> List[Dict[str, float]]:
        """
        Score multiple premises against a single hypothesis.
        Returns list of dicts with entail, neutral, contradict probabilities.
        """
        if not premises:
            return []

        results = []
        batch_size = 16

        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i + batch_size]

            # Tokenize premise-hypothesis pairs
            inputs = self.tokenizer(
                batch_premises,
                [hypothesis] * len(batch_premises),
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt"
            )

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for j in range(len(batch_premises)):
                results.append({
                    "entail": probs[j, self.entail_idx].item(),
                    "neutral": probs[j, self.neutral_idx].item(),
                    "contradict": probs[j, self.contradict_idx].item()
                })

        return results


def normalize_citation_tags(text: str) -> str:
    """Return a punctuation-preserving, sentence-normalized citation string."""
    return " ".join(
        f"{sentence} {''.join(tags)}".strip()
        for sentence, tags in _citation_sentences(text)
    )


def _clean_claim_text(text: str) -> str:
    return re.sub(r"\s+([.!?])", r"\1", text).strip()


def extract_claims(answer: str, use_llm: bool = False) -> List[Claim]:
    """
    Extract claims from answer.
    - Split into sentences with nltk
    - Strip citation tags such as [C1]
    - Drop sentences < 4 words and exact refusal
    """
    if not answer.strip():
        return []

    # Check for refusal
    refusal_phrases = [
        "do not contain enough information",
        "not contain enough information",
        "cannot answer",
        "unable to answer"
    ]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return []

    claims = []
    for clean_sent, citation_tags in _citation_sentences(answer):

        # Skip very short sentences (< 4 words)
        word_count = len(re.findall(r"[A-Za-z]+", clean_sent))
        if word_count < 3:
            continue

        # Skip refusal-like sentences
        if any(phrase in clean_sent.lower() for phrase in refusal_phrases):
            continue

        claims.append(Claim(clean_sent, [tag.strip("[]") for tag in citation_tags]))

    return claims


def _citation_sentences(text: str) -> List[Tuple[str, List[str]]]:
    """Split raw sentences without changing punctuation or merging claims."""
    records: List[List[Any]] = []
    citation_pattern = re.compile(r"\[C\d+\]")
    leading_pattern = re.compile(r"^\s*((?:\[C\d+\]\s*)+)")

    for raw_sentence in nltk.sent_tokenize(text):
        leading_match = leading_pattern.match(raw_sentence)
        leading_tags = (
            citation_pattern.findall(leading_match.group(1))
            if leading_match else []
        )
        body = raw_sentence[leading_match.end():] if leading_match else raw_sentence
        citation_tags = leading_tags + citation_pattern.findall(body)
        clean_sentence = re.sub(r"\s*\[C\d+\]", "", body).strip()
        word_count = len(re.findall(r"[A-Za-z]+", clean_sentence))

        if leading_tags and records:
            records[-1][1].extend(leading_tags)
        if word_count == 0:
            if not clean_sentence and citation_tags and records:
                records[-1][1].extend(
                    tag for tag in citation_tags if tag not in leading_tags
                )
            continue
        records.append([clean_sentence, citation_tags[len(leading_tags):]])

    return [(sentence, tags) for sentence, tags in records]


def split_into_windows(text: str, window_sentences: int = 3, window_stride: int = 1,
                       max_tokens: int = 400, tokenizer=None) -> List[str]:
    """Split text into sliding windows of sentences, truncated to max_tokens."""
    if not text.strip():
        return []

    sentences = nltk.sent_tokenize(text)

    if not sentences:
        return []

    windows = []
    for i in range(0, len(sentences), window_stride):
        window_sents = []
        token_count = 0
        for sentence in sentences[i:i + window_sentences]:
            sentence_tokens = (
                len(tokenizer.encode(sentence, add_special_tokens=False))
                if tokenizer else len(sentence.split())
            )
            if window_sents and token_count + sentence_tokens > max_tokens:
                break
            window_sents.append(sentence)
            token_count += sentence_tokens
        if window_sents:
            windows.append(" ".join(window_sents))
        if i + window_sentences >= len(sentences):
            break

    return windows


def verify_claim(claim: str, cited_chunks: List[Dict], all_chunks: List[Dict],
                 nli_scorer: NLIScorer, cfg: Dict,
                 relevance_scorer=None) -> Dict[str, Any]:
    """
    Verify a single claim against cited chunks first, then all chunks.
    Returns verdict with best evidence.
    """
    entail_threshold = cfg.get("entail_threshold", 0.5)
    contradict_threshold = cfg.get("contradict_threshold", 0.5)
    window_sentences = cfg.get("window_sentences", 3)
    window_stride = cfg.get("window_stride", 1)
    max_premise_tokens = cfg.get("max_premise_tokens", 400)
    top_windows = int(cfg.get("top_windows", 5))

    started = time.perf_counter()
    # First, check cited chunks
    best_result = {"label": "unverified", "entail": 0.0, "neutral": 0.0, "contradict": 0.0,
                   "best_chunk_id": None, "best_evidence": ""}

    normalized_claim = _normalize_for_match(claim)
    for chunk in cited_chunks:
        normalized_chunk = _normalize_for_match(chunk.get("text", ""))
        if normalized_claim and normalized_claim in normalized_chunk:
            evidence = _matching_sentence(chunk.get("text", ""), normalized_claim)
            return {
                **best_result,
                "label": "supported",
                "entail": 1.0,
                "best_chunk_id": chunk["chunk_id"],
                "best_evidence": evidence,
                "claim": claim,
                "nli_pairs_scored": 0,
                "nli_time": time.perf_counter() - started,
            }

    # Rank every retrieved evidence window against the claim before NLI.
    premises = []
    chunk_ids = []
    evidences = []
    for chunk in all_chunks:
        windows = split_into_windows(
            chunk["text"],
            window_sentences=window_sentences,
            window_stride=window_stride,
            max_tokens=max_premise_tokens,
            tokenizer=nli_scorer.tokenizer
        )
        for w in windows:
            premises.append(w)
            chunk_ids.append(chunk["chunk_id"])
            evidences.append(w)

    premises, chunk_ids, evidences = _dedupe_windows(premises, chunk_ids, evidences)
    relevance_scores = _score_relevance(claim, premises, relevance_scorer)
    ranked = sorted(
        zip(relevance_scores, premises, chunk_ids, evidences),
        key=lambda item: item[0],
        reverse=True,
    )[:top_windows]
    top_windows_detail = [
        {"chunk_id": chunk_id, "evidence": evidence, "score": float(score)}
        for score, _, chunk_id, evidence in ranked
    ]

    selected_premises = [item[1] for item in ranked]
    scores = nli_scorer.score(selected_premises, claim) if selected_premises else []
    pairs_scored = len(scores)
    if scores:
        max_entail_idx = max(range(len(scores)), key=lambda i: scores[i]["entail"])
        top_score = scores[0]
        if scores[max_entail_idx]["entail"] >= entail_threshold:
            verdict_idx = max_entail_idx
            label = "supported"
        elif (
            top_score["contradict"] >= contradict_threshold
            and top_score["contradict"] > top_score["entail"]
        ):
            verdict_idx = 0
            label = "unsupported"
        else:
            verdict_idx = 0
            label = "unverified"
        _, _, chunk_id, evidence = ranked[verdict_idx]
        best_result = {
            "label": label,
            "entail": scores[verdict_idx]["entail"],
            "neutral": scores[verdict_idx]["neutral"],
            "contradict": scores[verdict_idx]["contradict"],
            "best_chunk_id": chunk_id,
            "best_evidence": evidence,
        }

    best_result["claim"] = claim
    best_result["nli_pairs_scored"] = pairs_scored
    best_result["nli_time"] = time.perf_counter() - started
    best_result["top_windows"] = top_windows_detail
    return best_result


def _score_relevance(claim: str, premises: List[str], relevance_scorer) -> List[float]:
    """Score claim/window relevance in one cross-encoder batch."""
    if not premises:
        return []
    if relevance_scorer is not None:
        scores = relevance_scorer.predict(
            [(claim, premise) for premise in premises],
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
    claim_tokens = set(_normalize_for_match(claim).split())
    return [
        len(claim_tokens & set(_normalize_for_match(premise).split()))
        / max(len(claim_tokens), 1)
        for premise in premises
    ]


def _dedupe_windows(
    windows: List[str], chunk_ids: List[str], evidences: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    seen = set()
    result = []
    for window, chunk_id, evidence in zip(windows, chunk_ids, evidences):
        if window in seen:
            continue
        seen.add(window)
        result.append((window, chunk_id, evidence))
    if not result:
        return [], [], []
    values = list(zip(*result))
    return list(values[0]), list(values[1]), list(values[2])


def _normalize_for_match(text: str) -> str:
    text = re.sub(r"\[C\d+\]", "", text, flags=re.IGNORECASE)
    return " ".join(text.lower().split())


def _matching_sentence(text: str, normalized_claim: str) -> str:
    """Return the smallest sentence containing an exact-match claim."""
    for sentence in nltk.sent_tokenize(text):
        if normalized_claim in _normalize_for_match(sentence):
            return sentence.strip()
    return text.strip()


def _rerank_probability(chunk: Dict) -> float:
    if chunk.get("score_type") != "rerank":
        return 0.0
    score = float(chunk.get("score", 0.0))
    return 1.0 / (1.0 + torch.exp(torch.tensor(-score)).item())


def get_nli_scorer(model_name: str) -> NLIScorer:
    """Load one scorer per model name for non-Streamlit callers."""
    if model_name not in _NLI_SCORER_CACHE:
        _NLI_SCORER_CACHE[model_name] = NLIScorer(model_name)
    return _NLI_SCORER_CACHE[model_name]


def verify_answer(answer: str, citations: Dict[str, str], retrieved_chunks: List[Dict],
                  nli_scorer: NLIScorer, cfg: Dict, relevance_scorer=None) -> Dict[str, Any]:
    """
    Verify entire answer.
    Returns dict with claims, faithfulness score, counts, and overall flag.
    """
    # Check for abstention
    refusal_phrases = [
        "do not contain enough information",
        "not contain enough information",
        "cannot answer",
        "unable to answer"
    ]
    abstained = any(phrase in answer.lower() for phrase in refusal_phrases)

    if abstained:
        return {
            "claims": [],
            "faithfulness": None,
            "n_supported": 0,
            "n_unverified": 0,
            "n_unsupported": 0,
            "abstained": True,
            "overall_flag": "green"
        }

    # Extract claims
    claims = extract_claims(answer)

    if not claims:
        return {
            "claims": [],
            "faithfulness": 1.0,
            "n_supported": 0,
            "n_unverified": 0,
            "n_unsupported": 0,
            "abstained": False,
            "overall_flag": "green"
        }

    # Map citation tags to chunk_ids
    citation_to_chunk = citations  # e.g., {"C1": "chunk_id", ...}

    # Build chunk lookup
    chunk_by_id = {c["chunk_id"]: c for c in retrieved_chunks}

    # For each claim, find which chunks it cites
    citation_pattern = re.compile(r'\[C(\d+)\]')

    claim_verdicts = []

    for claim in claims:
        cited_chunk_ids = [
            citation_to_chunk.get(tag.strip("[]"))
            for tag in claim.citations
        ] or list(citation_to_chunk.values())
        cited_chunks = [chunk_by_id[cid] for cid in cited_chunk_ids if cid in chunk_by_id]

        verdict = verify_claim(
            claim.text, cited_chunks, retrieved_chunks, nli_scorer, cfg,
            relevance_scorer=relevance_scorer,
        )
        verdict["citations"] = claim.citations
        claim_verdicts.append(verdict)

    # Count verdicts
    n_supported = sum(1 for v in claim_verdicts if v["label"] == "supported")
    n_unverified = sum(1 for v in claim_verdicts if v["label"] == "unverified")
    n_unsupported = sum(1 for v in claim_verdicts if v["label"] == "unsupported")

    total = len(claim_verdicts)
    faithfulness = n_supported / total if total > 0 else 1.0

    # Overall flag
    if n_unsupported > 0:
        overall_flag = "red"
    elif n_unverified > 0:
        overall_flag = "amber"
    else:
        overall_flag = "green"

    return {
        "claims": claim_verdicts,
        "faithfulness": faithfulness,
        "n_supported": n_supported,
        "n_unverified": n_unverified,
        "n_unsupported": n_unsupported,
        "abstained": False,
        "overall_flag": overall_flag
        ,"nli_pairs_scored": sum(v.get("nli_pairs_scored", 0) for v in claim_verdicts)
        ,"claim_timings": [
            {"claim": v["claim"], "pairs": v.get("nli_pairs_scored", 0),
             "seconds": v.get("nli_time", 0.0)}
            for v in claim_verdicts
        ]
    }


if __name__ == "__main__":
    # Quick test with fixed cases
    print("=== Faithfulness Module Test ===")

    # Load NLI model
    nli_model = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    print(f"Loading NLI model: {nli_model}...")
    scorer = NLIScorer(nli_model)
    print(f"Label mapping: {scorer.id2label}")
    print(f"Entail idx: {scorer.entail_idx}, Neutral: {scorer.neutral_idx}, Contradict: {scorer.contradict_idx}")

    cfg = {
        "entail_threshold": 0.5,
        "contradict_threshold": 0.5,
        "window_sentences": 3,
        "window_stride": 1,
        "max_premise_tokens": 400
    }

    # Test cases
    test_cases = [
        {
            "premise": "The model was trained on 10,000 English documents.",
            "claim": "The model used 10,000 English documents for training.",
            "expected": "supported"
        },
        {
            "premise": "The model was trained on 10,000 English documents.",
            "claim": "The model was trained on French documents.",
            "expected": "unsupported"
        },
        {
            "premise": "The model was trained on 10,000 English documents.",
            "claim": "The model achieved state-of-the-art accuracy.",
            "expected": "unverified"
        }
    ]

    print("\n--- Test Cases ---")
    for tc in test_cases:
        premises = [tc["premise"]]
        claim = tc["claim"]

        scores = scorer.score(premises, claim)
        result = scores[0]

        entail = result["entail"]
        contradict = result["contradict"]

        if entail >= 0.5:
            label = "supported"
        elif contradict >= 0.5:
            label = "unsupported"
        else:
            label = "unverified"

        status = "✓" if label == tc["expected"] else "✗"
        print(f"{status} Premise: {tc['premise']}")
        print(f"    Claim: {claim}")
        print(f"    Entail: {entail:.3f}, Contradict: {contradict:.3f} -> {label} (expected: {tc['expected']})")
        print()

    print("Done!")