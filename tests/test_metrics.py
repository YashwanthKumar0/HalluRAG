"""Tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation import (
    token_overlap,
    is_relevant,
    recall_at_k,
    mrr,
    token_f1,
    answer_correctness,
    is_abstained,
    hallucination_rate,
    unsupported_claims_rate,
    abstention_accuracy
)


class TestTokenOverlap:
    def test_identical_texts(self):
        text = "the quick brown fox"
        assert token_overlap(text, text) == 1.0

    def test_no_overlap(self):
        assert token_overlap("a b c", "d e f") == 0.0

    def test_partial_overlap(self):
        # "a b c" and "b c d" -> intersection {b,c}, union {a,b,c,d} -> 2/4 = 0.5
        assert token_overlap("a b c", "b c d") == 0.5

    def test_case_insensitive(self):
        assert token_overlap("A B C", "a b c") == 1.0

    def test_empty_strings(self):
        assert token_overlap("", "a b c") == 0.0
        assert token_overlap("a b c", "") == 0.0


class TestIsRelevant:
    def setup_method(self):
        self.gold = {
            "paper_id": "paper1",
            "supporting_text": "X is Y according to the paper."
        }

    def test_same_paper_high_overlap(self):
        chunk = {
            "paper_id": "paper1",
            "text": "X is Y according to the paper. Some more text."
        }
        assert is_relevant(chunk, self.gold) == True

    def test_different_paper(self):
        chunk = {
            "paper_id": "paper2",
            "text": "X is Y according to the paper."
        }
        assert is_relevant(chunk, self.gold) == False

    def test_same_paper_low_overlap(self):
        chunk = {
            "paper_id": "paper1",
            "text": "Completely different text here."
        }
        assert is_relevant(chunk, self.gold) == False

    def test_empty_supporting_text(self):
        gold = {"paper_id": "paper1", "supporting_text": ""}
        chunk = {"paper_id": "paper1", "text": "Some text"}
        assert is_relevant(chunk, gold) == False


class TestRecallAtK:
    def setup_method(self):
        self.gold = {
            "paper_id": "paper1",
            "supporting_text": "X is Y according to the paper."
        }
        self.chunks = [
            {"paper_id": "paper1", "text": "X is Y according to the paper."},
            {"paper_id": "paper2", "text": "Other text."},
            {"paper_id": "paper1", "text": "More text from paper1."}
        ]

    def test_recall_at_1_hit(self):
        assert recall_at_k(self.chunks, self.gold, 1) == 1.0

    def test_recall_at_1_miss(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other text."},
            {"paper_id": "paper1", "text": "X is Y according to the paper."}
        ]
        assert recall_at_k(chunks, self.gold, 1) == 0.0

    def test_recall_at_3(self):
        assert recall_at_k(self.chunks, self.gold, 3) == 1.0

    def test_recall_increases_with_k(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other text."},
            {"paper_id": "paper1", "text": "X is Y according to the paper."},
        ]
        assert recall_at_k(chunks, self.gold, 1) < recall_at_k(chunks, self.gold, 10)

    def test_recall_no_hit(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other text."},
            {"paper_id": "paper3", "text": "More other text."}
        ]
        assert recall_at_k(chunks, self.gold, 5) == 0.0


class TestMRR:
    def setup_method(self):
        self.gold = {
            "paper_id": "paper1",
            "supporting_text": "X is Y according to the paper."
        }

    def test_mrr_first(self):
        chunks = [
            {"paper_id": "paper1", "text": "X is Y according to the paper."},
            {"paper_id": "paper2", "text": "Other."}
        ]
        assert mrr(chunks, self.gold) == 1.0

    def test_mrr_second(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other."},
            {"paper_id": "paper1", "text": "X is Y according to the paper."}
        ]
        assert mrr(chunks, self.gold) == 0.5

    def test_mrr_third(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other."},
            {"paper_id": "paper3", "text": "More."},
            {"paper_id": "paper1", "text": "X is Y according to the paper."}
        ]
        assert mrr(chunks, self.gold) == 1.0 / 3.0

    def test_mrr_none(self):
        chunks = [
            {"paper_id": "paper2", "text": "Other."},
            {"paper_id": "paper3", "text": "More."}
        ]
        assert mrr(chunks, self.gold) == 0.0


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("X is Y", "X is Y") == 1.0

    def test_partial_match(self):
        # "X is Y and Z" vs "X is Y" -> common: X, is, Y (3)
        # pred: 5 tokens, gt: 3 tokens
        # precision = 3/5, recall = 3/3 = 1
        # F1 = 2 * (3/5 * 1) / (3/5 + 1) = 2 * 0.6 / 1.6 = 0.75
        f1 = token_f1("X is Y and Z", "X is Y")
        assert abs(f1 - 0.75) < 0.01

    def test_no_match(self):
        assert token_f1("A B C", "X Y Z") == 0.0

    def test_empty_strings(self):
        assert token_f1("", "X Y") == 0.0
        assert token_f1("X Y", "") == 0.0

    def test_case_insensitive(self):
        assert token_f1("X IS Y", "x is y") == 1.0


class TestAnswerCorrectness:
    def test_same_as_token_f1(self):
        assert answer_correctness("X is Y", "X is Y") == 1.0
        # "X is Y" vs "A is B" shares "is" -> 1/3 = 0.333...
        assert answer_correctness("X is Y", "A is B") == 1/3
        assert answer_correctness("X is Y", "A B C") == 0.0


class TestIsAbstained:
    def test_standard_refusal(self):
        assert is_abstained("The provided documents do not contain enough information to answer this question.") == True

    def test_variants(self):
        assert is_abstained("The documents do not contain enough information.") == True
        assert is_abstained("I cannot answer this question.") == True
        assert is_abstained("Unable to answer.") == True

    def test_normal_answers(self):
        assert is_abstained("X is Y.") == False
        assert is_abstained("The answer is 42.") == False


class TestHallucinationRate:
    def test_basic(self):
        answers = [
            {"overall_flag": "red"},
            {"overall_flag": "green"},
            {"overall_flag": "amber"},
            {"overall_flag": "red"}
        ]
        assert hallucination_rate(answers) == 0.5

    def test_no_red(self):
        answers = [
            {"overall_flag": "green"},
            {"overall_flag": "amber"}
        ]
        assert hallucination_rate(answers) == 0.0

    def test_all_red(self):
        answers = [
            {"overall_flag": "red"},
            {"overall_flag": "red"}
        ]
        assert hallucination_rate(answers) == 1.0

    def test_empty(self):
        assert hallucination_rate([]) == 0.0


class TestUnsupportedClaimsRate:
    def test_basic(self):
        answers = [
            {"claims": [{"label": "supported"}, {"label": "unsupported"}]},
            {"claims": [{"label": "supported"}]}
        ]
        # 1 unsupported out of 3 total
        assert abs(unsupported_claims_rate(answers) - 1/3) < 0.01

    def test_no_claims(self):
        answers = [
            {"claims": []},
            {"claims": []}
        ]
        assert unsupported_claims_rate(answers) == 0.0


class TestAbstentionAccuracy:
    def test_basic(self):
        gold = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": None}  # unanswerable
        ]
        answers = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "The provided documents do not contain enough information."}
        ]
        assert abstention_accuracy(answers, gold) == 1.0

    def test_failed_abstention(self):
        gold = [
            {"question": "Q1", "answer": None}
        ]
        answers = [
            {"question": "Q1", "answer": "Some guess answer."}
        ]
        assert abstention_accuracy(answers, gold) == 0.0

    def test_no_unanswerable(self):
        gold = [
            {"question": "Q1", "answer": "A1"}
        ]
        answers = [
            {"question": "Q1", "answer": "A1"}
        ]
        assert abstention_accuracy(answers, gold) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])