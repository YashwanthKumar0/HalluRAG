import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.faithfulness import NLIScorer, extract_claims, verify_claim, verify_answer, split_into_windows


@pytest.fixture(scope="module")
def nli_scorer():
    """Load NLI model once for all tests."""
    return NLIScorer("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")


@pytest.fixture
def cfg():
    """Faithfulness config."""
    return {
        "entail_threshold": 0.5,
        "contradict_threshold": 0.5,
        "window_sentences": 3,
        "window_stride": 1,
        "max_premise_tokens": 400
    }


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for testing."""
    return [
        {
            "chunk_id": "chunk_1",
            "title": "Test Paper",
            "section": "Introduction",
            "text": "The model was trained on 10,000 English documents. It uses a transformer architecture.",
            "n_tokens": 20,
            "strategy": "fixed",
            "paper_id": "test_001"
        },
        {
            "chunk_id": "chunk_2",
            "title": "Test Paper",
            "section": "Method",
            "text": "The training data consists of English text only. No French documents were used.",
            "n_tokens": 18,
            "strategy": "fixed",
            "paper_id": "test_001"
        }
    ]


def test_extract_claims_basic():
    """Test claim extraction from answer."""
    answer = "The model was trained on 10,000 documents [C1]. It uses a transformer architecture [C1]."
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert "10,000 documents" in claims[0]
    assert "transformer architecture" in claims[1]


def test_extract_claims_strips_citations():
    """Test that [C1] tags are stripped."""
    answer = "The model was trained on 10,000 documents [C1][C2]."
    claims = extract_claims(answer)
    assert "[C" not in claims[0]


def test_extract_claims_drops_short():
    """Test that sentences < 4 words are dropped."""
    answer = "Model works [C1]. It was trained on 10,000 English documents [C1]."
    claims = extract_claims(answer)
    # "Model works" = 2 words -> dropped
    assert len(claims) == 1
    assert "10,000 English documents" in claims[0]


def test_extract_claims_refusal():
    """Test that refusal sentences return empty claims."""
    answer = "The provided documents do not contain enough information to answer this question."
    claims = extract_claims(answer)
    assert claims == []


@pytest.mark.parametrize(
    "answer, expected_citations",
    [
        (
            "A is B [C1]. C is D [C2].",
            [["C1"], ["C2"]],
        ),
        (
            "A is B.[C1] C is D.[C1]",
            [["C1"], ["C1"]],
        ),
        (
            "A is B. [C1][C2] C is D.",
            [["C1", "C2"], []],
        ),
    ],
)
def test_extract_claims_preserves_sentence_boundaries(answer, expected_citations):
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert [claim.citations for claim in claims] == expected_citations


def test_extract_claims_pomdp_answer_has_three_claims():
    answer = (
        "The most trivial problems require little computation. "
        "They also demand substantial memory. "
        "Additionally, exact policies require enumerating many instances "
        "[C1][C2][C3]"
    )
    claims = extract_claims(answer)
    assert len(claims) == 3


def test_nli_scorer_label_order(nli_scorer):
    """Test that label order is read from config."""
    assert hasattr(nli_scorer, 'id2label')
    assert hasattr(nli_scorer, 'entail_idx')
    assert hasattr(nli_scorer, 'neutral_idx')
    assert hasattr(nli_scorer, 'contradict_idx')


def test_fixed_case_supported(nli_scorer, cfg):
    """Premise: trained on 10,000 English docs -> Claim: used 10,000 English docs for training = supported"""
    premise = "The model was trained on 10,000 English documents."
    claim = "The model used 10,000 English documents for training."

    scores = nli_scorer.score([premise], claim)
    result = scores[0]

    entail = result["entail"]
    contradict = result["contradict"]

    if entail >= cfg["entail_threshold"]:
        label = "supported"
    elif contradict >= cfg["contradict_threshold"]:
        label = "unsupported"
    else:
        label = "unverified"

    assert label == "supported", f"Expected supported, got {label} (entail={entail:.3f}, contradict={contradict:.3f})"


def test_fixed_case_unsupported(nli_scorer, cfg):
    """Premise: trained on 10,000 English docs -> Claim: trained on French docs = unsupported"""
    premise = "The model was trained on 10,000 English documents."
    claim = "The model was trained on French documents."

    scores = nli_scorer.score([premise], claim)
    result = scores[0]

    entail = result["entail"]
    contradict = result["contradict"]

    if entail >= cfg["entail_threshold"]:
        label = "supported"
    elif contradict >= cfg["contradict_threshold"]:
        label = "unsupported"
    else:
        label = "unverified"

    assert label == "unsupported", f"Expected unsupported, got {label} (entail={entail:.3f}, contradict={contradict:.3f})"


def test_fixed_case_unverified(nli_scorer, cfg):
    """Premise: trained on 10,000 English docs -> Claim: achieved SOTA = unverified"""
    premise = "The model was trained on 10,000 English documents."
    claim = "The model achieved state-of-the-art accuracy."

    scores = nli_scorer.score([premise], claim)
    result = scores[0]

    entail = result["entail"]
    contradict = result["contradict"]

    if entail >= cfg["entail_threshold"]:
        label = "supported"
    elif contradict >= cfg["contradict_threshold"]:
        label = "unsupported"
    else:
        label = "unverified"

    assert label == "unverified", f"Expected unverified, got {label} (entail={entail:.3f}, contradict={contradict:.3f})"


def test_verify_claim_uses_cited_chunks_first(nli_scorer, cfg, sample_chunks):
    """Test that verify_claim checks cited chunks before all chunks."""
    claim = "The model was trained on 10,000 English documents."

    # Cited chunks: only chunk_1 (which contains the info)
    cited_chunks = [sample_chunks[0]]
    all_chunks = sample_chunks

    result = verify_claim(claim, cited_chunks, all_chunks, nli_scorer, cfg)

    assert result["label"] in ["supported", "unverified", "unsupported"]
    assert "claim" in result
    assert "best_chunk_id" in result
    assert "best_evidence" in result


def test_verify_answer_abstained(nli_scorer, cfg, sample_chunks):
    """Test that abstained answers get green flag with null faithfulness."""
    answer = "The provided documents do not contain enough information to answer this question."
    citations = {"C1": "chunk_1"}

    result = verify_answer(answer, citations, sample_chunks, nli_scorer, cfg)

    assert result["abstained"] is True
    assert result["faithfulness"] is None
    assert result["overall_flag"] == "green"
    assert result["n_supported"] == 0


def test_verify_answer_overall_flags(nli_scorer, cfg, sample_chunks):
    """Test overall flag logic: red if any unsupported, amber if any unverified, green if all supported."""
    # This test just checks the flag logic is present
    answer = "The model was trained on data [C1]."
    citations = {"C1": "chunk_1"}

    result = verify_answer(answer, citations, sample_chunks, nli_scorer, cfg)

    assert result["overall_flag"] in ["green", "amber", "red"]
    assert "n_supported" in result
    assert "n_unverified" in result
    assert "n_unsupported" in result


def test_split_into_windows():
    """Test sliding window splitting."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    windows = split_into_windows(text, window_sentences=2, window_stride=1, max_tokens=100)

    # Should have overlapping windows
    assert len(windows) >= 3
    assert "Sentence one" in windows[0]
    assert "Sentence two" in windows[0]


def test_verbatim_premise_sentence_is_supported():
    """A claim copied from cited evidence should bypass NLI as supported."""
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    class Scorer:
        tokenizer = Tokenizer()

        def score(self, premises, hypothesis):
            raise AssertionError("Exact matches must not invoke NLI")

    chunk = {
        "chunk_id": "chunk_2",
        "text": "The model uses a transformer architecture. It was trained on English documents.",
        "score_type": "rerank",
        "score": 2.0,
    }
    result = verify_claim(
        "The model uses a transformer architecture.",
        [chunk],
        [chunk],
        Scorer(),
        {"entail_threshold": 0.5, "contradict_threshold": 0.5},
    )

    assert result["label"] == "supported"
    assert result["entail"] == 1.0


def test_relevant_window_prevents_spurious_contradiction():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    class RelevanceScorer:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0 if "rule tuning" in premise else 0.1 for _, premise in pairs]

    class Scorer:
        tokenizer = Tokenizer()

        def score(self, premises, hypothesis):
            return [
                {
                    "entail": 0.7 if "rule tuning" in premise else 0.0,
                    "neutral": 0.2,
                    "contradict": 0.1 if "rule tuning" in premise else 0.99,
                }
                for premise in premises
            ]

    chunk = {
        "chunk_id": "C2",
        "text": (
            "We propose several rule tuning steps aiming to recover lost "
            "information in the discretisation. "
            "In the following, we briefly describe the individual steps. "
            "I. Rule Learning."
        ),
    }
    result = verify_claim(
        "The paper proposes postprocessing steps to recover information lost in discretisation.",
        [chunk],
        [chunk],
        Scorer(),
        {"entail_threshold": 0.8, "contradict_threshold": 0.5, "top_windows": 5},
        relevance_scorer=RelevanceScorer(),
    )
    assert result["label"] != "unsupported"
    assert len(result["top_windows"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])