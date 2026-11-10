#!/usr/bin/env python
"""
Draft gold QA set for evaluation.

1. Randomly sample 60 chunks (semantic strategy, seed 42), spread across ≥ 30 papers.
2. For each, ask LLM: "Write one specific factual question answerable ONLY from this passage,
   the short answer, and copy the exact sentence(s) that contain the answer.
   Return JSON with keys question, answer, supporting_text."
3. Validate: supporting_text must appear verbatim (after whitespace normalisation) in the chunk;
   otherwise discard.
4. Write data/gold/gold_qa_draft.jsonl.
5. Print instructions for human review.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.utils import read_jsonl, write_jsonl


SYSTEM_PROMPT = """Return JSON with keys question, answer, supporting_text. Be concise."""


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison."""
    return re.sub(r'\s+', ' ', text).strip()


def supporting_text_in_chunk(supporting_text: str, chunk_text: str) -> bool:
    """Check if supporting_text appears in chunk_text after whitespace normalization."""
    norm_support = normalize_whitespace(supporting_text)
    norm_chunk = normalize_whitespace(chunk_text)
    return norm_support in norm_chunk


def call_llm(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            if content is None:
                continue

            # Try to parse JSON from response
            content = content.strip()
            # Handle potential code block wrapping
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            # Validate required keys
            if all(k in data for k in ("question", "answer", "supporting_text")):
                return data

        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"  Failed after {max_retries} attempts: {e}")
            continue

    return None


def sample_chunks(chunks_path: str, n_samples: int = 60, min_papers: int = 30, seed: int = 42) -> List[Dict]:
    """Sample chunks spread across papers."""
    random.seed(seed)

    chunks = read_jsonl(chunks_path)
    print(f"Loaded {len(chunks)} semantic chunks")

    # Group by paper_id
    by_paper = {}
    for chunk in chunks:
        pid = chunk["paper_id"]
        if pid not in by_paper:
            by_paper[pid] = []
        by_paper[pid].append(chunk)

    print(f"Chunks from {len(by_paper)} papers")

    # Ensure we have enough papers
    if len(by_paper) < min_papers:
        print(f"WARNING: Only {len(by_paper)} papers available, need at least {min_papers}")

    # Sample papers first
    paper_ids = list(by_paper.keys())
    random.shuffle(paper_ids)
    selected_papers = paper_ids[:min_papers]

    # Now sample chunks from selected papers
    selected_chunks = []
    for pid in selected_papers:
        paper_chunks = by_paper[pid]
        # Take 1-3 chunks per paper
        n_from_paper = min(len(paper_chunks), random.randint(1, 3))
        sampled = random.sample(paper_chunks, n_from_paper)
        selected_chunks.extend(sampled)

        if len(selected_chunks) >= n_samples:
            break

    # If we still need more, sample from remaining papers
    if len(selected_chunks) < n_samples:
        remaining_chunks = [c for c in chunks if c["paper_id"] not in selected_papers]
        random.shuffle(remaining_chunks)
        selected_chunks.extend(remaining_chunks[:n_samples - len(selected_chunks)])

    selected_chunks = selected_chunks[:n_samples]
    print(f"Selected {len(selected_chunks)} chunks from {len(set(c['paper_id'] for c in selected_chunks))} papers")

    return selected_chunks


def main():
    parser = argparse.ArgumentParser(description="Draft gold QA set")
    parser.add_argument("--n-samples", type=int, default=60, help="Number of chunks to sample")
    parser.add_argument("--min-papers", type=int, default=30, help="Minimum papers to spread across")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/gold/gold_qa_draft.jsonl", help="Output path")
    args = parser.parse_args()

    config = load_config()

    # Get LLM config
    gen_cfg = config.get("generation", {})
    base_url = gen_cfg.get("base_url")
    api_key = gen_cfg.get("api_key")
    model = gen_cfg.get("model")

    if not base_url or not api_key or not model:
        print("ERROR: Generation config missing base_url, api_key, or model")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)

    # Sample chunks
    chunks_path = Path(config["data"]["chunks_dir"]) / "chunks_semantic.jsonl"
    chunks = sample_chunks(str(chunks_path), args.n_samples, args.min_papers, args.seed)

    # Generate QA pairs
    qa_items = []

    for i, chunk in enumerate(tqdm(chunks, desc="Generating QA pairs")):
        prompt = f"""Passage (from paper: {chunk['title']}, section: {chunk['section']}):
---
{chunk['text']}
---

Create one factual question answerable ONLY from this passage. Return JSON with keys: question, answer, supporting_text."""

        result = call_llm(client, model, prompt)

        if result is None:
            print(f"  Failed to generate for chunk {chunk['chunk_id']}")
            continue

        # Validate supporting_text appears in chunk
        if not supporting_text_in_chunk(result["supporting_text"], chunk["text"]):
            print(f"  WARNING: supporting_text not found in chunk {chunk['chunk_id']}")
            support_preview = result['supporting_text'][:100].encode('ascii', 'replace').decode('ascii')
            chunk_preview = chunk['text'][:200].encode('ascii', 'replace').decode('ascii')
            print(f"    supporting_text: {support_preview}...")
            print(f"    chunk text (first 200): {chunk_preview}...")
            continue

        # Build QA item
        qa_item = {
            "question": result["question"],
            "answer": result["answer"],
            "supporting_text": result["supporting_text"],
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "title": chunk["title"],
            "section": chunk["section"]
        }
        qa_items.append(qa_item)
        question_preview = result['question'][:80].encode('ascii', 'replace').decode('ascii')
        print(f"  [{i+1}/{len(chunks)}] {chunk['chunk_id']}: {question_preview}...")

    # Write draft
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), qa_items)

    print(f"\n{'='*60}")
    print(f"DRAFT COMPLETE: {len(qa_items)} items written to {output_path}")
    print(f"{'='*60}")
    print()
    print("NEXT STEPS FOR HUMAN REVIEW:")
    print("1. Open data/gold/gold_qa_draft.jsonl")
    print("2. Review each item:")
    print("   - Check question is specific and answerable ONLY from the passage")
    print("   - Check answer is correct and concise")
    print("   - Check supporting_text is verbatim from the chunk")
    print("   - Remove vague questions (e.g., 'What is this paper about?')")
    print("   - Remove questions answerable without the paper (general knowledge)")
    print("   - Edit questions/answers as needed")
    print("3. Save the final 30-40 items as data/gold/gold_qa.jsonl")
    print("4. ADD 3-5 UNANSWERABLE QUESTIONS by hand (answer: null) to test abstention, e.g.:")
    print('   {"question": "What is the population of France in 2024?", "answer": null, "supporting_text": "", "chunk_id": "", "paper_id": "", "title": "", "section": "unanswerable"}')
    print("   (These should be completely unrelated to any paper in the corpus)")


if __name__ == "__main__":
    main()