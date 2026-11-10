#!/usr/bin/env python
"""
Print a comprehensive corpus report: paper count by year, by primary category,
% of chunks with section == "Body", and 20 random titles.
"""
import json
import random
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.utils import read_jsonl


def arxiv_id_to_year(paper_id: str) -> int:
    """Extract year from arXiv ID. Handles new-style id "YYMM.NNNNN" -> year 2000+YY."""
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper_id):
        year_part = int(paper_id[:2])
        return 2000 + year_part if year_part < 50 else 1900 + year_part
    if '/' in paper_id:
        _, rest = paper_id.split('/', 1)
        if re.match(r'^\d{7}', rest):
            year_part = int(rest[:2])
            return 2000 + year_part if year_part < 50 else 1900 + year_part
    return None


import re


def is_primary_category_match(categories_str: str, allowed_categories: set) -> bool:
    """Check if the PRIMARY category (first in list) is in allowed_categories."""
    cats = categories_str.strip().split()
    if not cats:
        return False
    primary = cats[0]
    return primary in allowed_categories


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    chunks_dir = Path(config["data"]["chunks_dir"])

    print("=" * 70)
    print("CORPUS REPORT")
    print("=" * 70)

    # Load metadata
    metadata_path = raw_dir / "metadata_subset.jsonl"
    if metadata_path.exists():
        print("\nLoading metadata from:", metadata_path)
        metadata = read_jsonl(metadata_path)
    else:
        print("No metadata file found. Run scripts/01_download_data.py first.")
        return

    print(f"\nTotal metadata records: {len(metadata)}")

    # Filter by primary category and year
    categories = set(config["data"]["categories"])
    year_min = config["data"]["year_min"]
    keywords = config["data"].get("keywords", [])

    filtered = []
    for m in metadata:
        if not is_primary_category_match(m.get("categories", ""), categories):
            continue
        year = arxiv_id_to_year(m["paper_id"])
        if year is None or year < year_min:
            continue
        filtered.append(m)

    print(f"After primary category + year filter: {len(filtered)}")

    # Paper count by year
    years = [arxiv_id_to_year(m["paper_id"]) for m in filtered]
    year_counts = Counter(y for y in years if y is not None)
    print("\nPapers by year:")
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]}")

    # Paper count by primary category
    primary_cats = [m["categories"].strip().split()[0] for m in filtered if m.get("categories")]
    cat_counts = Counter(primary_cats)
    print("\nPapers by primary category:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # 20 random titles
    print("\n20 random titles (seed=0):")
    random.seed(0)
    for m in random.sample(filtered, min(20, len(filtered))):
        print(f"  {m['paper_id']}: {m['title'][:80]}...")

    # Check processed papers
    if processed_dir.exists():
        papers = list(processed_dir.glob("*.json"))
        print(f"\n\nProcessed papers: {len(papers)}")

        # Year range from processed
        proc_years = []
        for p in papers:
            pid = p.stem
            if re.match(r'^\d{4}\.', pid):
                year = int(pid[:2])
                proc_years.append(2000 + year if year < 50 else 1900 + year)
        if proc_years:
            print(f"Year range (processed): {min(proc_years)}–{max(proc_years)}")

    # Check chunks
    print("\n\nChunk analysis:")
    for strategy in ["fixed", "semantic"]:
        chunks_path = chunks_dir / f"chunks_{strategy}.jsonl"
        if chunks_path.exists():
            chunks = read_jsonl(str(chunks_path))
            print(f"\n{strategy.upper()} chunks: {len(chunks)}")

            # Section distribution
            section_counts = Counter(c.get("section", "Body") for c in chunks)
            body_count = section_counts.get("Body", 0)
            body_pct = (body_count / len(chunks) * 100) if chunks else 0
            print(f"  Sections:")
            for sec, cnt in section_counts.most_common(10):
                print(f"    {sec}: {cnt}")
            print(f"  % 'Body' sections: {body_pct:.1f}%")
            if body_pct > 50:
                print(f"  ⚠️  WARNING: More than 50% of chunks have section 'Body'!")

            # Paper coverage
            paper_ids = set(c.get("paper_id") for c in chunks)
            print(f"  Unique papers covered: {len(paper_ids)}")

            # Average chunk length
            avg_tokens = sum(c.get("n_tokens", 0) for c in chunks) / len(chunks) if chunks else 0
            print(f"  Avg tokens per chunk: {avg_tokens:.1f}")

    print("\n" + "=" * 70)
    print("HEALTHY CORPUS INDICATORS:")
    print("  - Papers from 2022+ (year_min)")
    print("  - Primary categories only (cs.CL, cs.AI)")
    print("  - < 50% chunks with section 'Body'")
    print("  - ~500-600 papers, ~5000-10000 chunks per strategy")
    print("  - Section names like 'Introduction', 'Method', 'Results', etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()