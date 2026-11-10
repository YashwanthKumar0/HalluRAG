#!/usr/bin/env python
"""
Script 4: Draft a gold Q&A set for evaluation.
This creates a template that can be manually filled with questions, answers, and relevant chunks.
Run after 03_build_indexes.py
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_config
from src.utils import setup_logging, load_jsonl, ensure_dir, save_jsonl
from src.indexing import load_chunks


def main():
    logger = setup_logging()
    logger.info("Drafting gold Q&A set...")

    cfg = get_config()

    gold_dir = Path(cfg.data.get("gold_dir", "data/gold"))
    ensure_dir(gold_dir)

    # Load chunks to get paper info
    chunks_fixed = load_chunks(Path(cfg.data.get("chunks_dir", "data/chunks")), "fixed")
    chunks_semantic = load_chunks(Path(cfg.data.get("chunks_dir", "data/chunks")), "semantic")

    # Get unique papers
    papers = {}
    for c in chunks_fixed:
        papers[c["paper_id"]] = {
            "paper_id": c["paper_id"],
            "paper_title": c["paper_title"],
        }

    logger.info(f"Found {len(papers)} papers")

    # Create template gold Q&A
    # This is a template - users should edit this file to add real questions
    template_questions = [
        {
            "question_id": "qa_001",
            "question": "What method does the paper propose for X?",
            "answer": "The paper proposes...",
            "relevant_chunks": [],  # Fill with chunk_ids from chunks_fixed or chunks_semantic
            "paper_ids": [],  # Optional: relevant paper IDs
            "category": "cs.CL",  # or cs.AI
            "difficulty": "easy",  # easy, medium, hard
        },
        {
            "question_id": "qa_002",
            "question": "How does the proposed method compare to baselines?",
            "answer": "The method outperforms baselines by...",
            "relevant_chunks": [],
            "paper_ids": [],
            "category": "cs.CL",
            "difficulty": "medium",
        },
        {
            "question_id": "qa_003",
            "question": "What are the limitations of the approach?",
            "answer": "The limitations include...",
            "relevant_chunks": [],
            "paper_ids": [],
            "category": "cs.AI",
            "difficulty": "hard",
        },
    ]

    gold_qa_path = gold_dir / "gold_qa.jsonl"

    if gold_qa_path.exists():
        logger.info(f"Gold Q&A already exists at {gold_qa_path}")
        logger.info("Please edit it to add your evaluation questions.")
        existing = load_jsonl(gold_qa_path)
        logger.info(f"Existing questions: {len(existing)}")
    else:
        save_jsonl(template_questions, gold_qa_path)
        logger.info(f"Created template at {gold_qa_path}")
        logger.info("IMPORTANT: Edit this file to add real evaluation questions!")
        logger.info("You need to:")
        logger.info("  1. Replace template questions with real questions about the papers")
        logger.info("  2. Add ground truth answers")
        logger.info("  3. Find relevant chunk IDs from the chunks files")
        logger.info("  4. Set relevant_chunks for each question")

    # Also create human labels template
    human_labels_path = gold_dir / "human_labels.jsonl"
    if not human_labels_path.exists():
        template_labels = [
            {
                "question_id": "qa_001",
                "faithfulness_label": "supported",  # supported, unverified, unsupported
                "notes": "",
            },
            {
                "question_id": "qa_002",
                "faithfulness_label": "supported",
                "notes": "",
            },
            {
                "question_id": "qa_003",
                "faithfulness_label": "unverified",
                "notes": "",
            },
        ]
        save_jsonl(template_labels, human_labels_path)
        logger.info(f"Created human labels template at {human_labels_path}")
        logger.info("Fill in faithfulness labels after running experiments")


if __name__ == "__main__":
    main()