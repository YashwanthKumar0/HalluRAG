#!/usr/bin/env python
"""
Download arXiv metadata and PDFs, extract and clean text.
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.data_prep import extract_paper
from src.utils import write_jsonl, append_jsonl


def arxiv_id_to_year(paper_id: str) -> Optional[int]:
    """Extract year from arXiv ID. Handles old (cs.CL/0001001) and new (YYMM.NNNNN) formats."""
    # New format: YYMM.NNNNN or YYMM.NNNNNvX
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper_id):
        year_part = int(paper_id[:2])
        return 2000 + year_part if year_part < 50 else 1900 + year_part
    # Old format: archive/YYMMNNN
    if '/' in paper_id:
        _, rest = paper_id.split('/', 1)
        if re.match(r'^\d{7}', rest):
            year_part = int(rest[:2])
            return 2000 + year_part if year_part < 50 else 1900 + year_part
    return None


def is_primary_category_match(categories_str: str, allowed_categories: set) -> bool:
    """Check if the PRIMARY category (first in list) is in allowed_categories."""
    cats = categories_str.strip().split()
    if not cats:
        return False
    primary = cats[0]
    return primary in allowed_categories


def matches_keywords(title: str, abstract: str, keywords: List[str]) -> bool:
    """Check if title or abstract contains any keyword (case-insensitive)."""
    if not keywords:
        return True
    text = (title + " " + abstract).lower()
    return any(kw.lower() in text for kw in keywords)


import re  # Move import to top in real file, keeping here for completeness


def load_kaggle_metadata(config: Dict, limit: int) -> List[Dict]:
    """Stream Kaggle metadata file line by line."""
    kaggle_path = Path(config["data"]["raw_dir"]) / "arxiv-metadata-oai-snapshot.json"
    if not kaggle_path.exists():
        return []

    categories = set(config["data"]["categories"])
    year_min = config["data"]["year_min"]
    max_papers = config["data"]["max_papers"]
    buffer_limit = int(max_papers * 1.3)
    keywords = config["data"].get("keywords", [])

    results = []
    dropped_year = 0
    dropped_primary_cat = 0
    dropped_keywords = 0
    with open(kaggle_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Kaggle metadata"):
            if len(results) >= buffer_limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_id = record.get("id", "")
            title = record.get("title", "").strip()
            abstract = record.get("abstract", "").strip()
            cats = record.get("categories", "").strip()

            # Check PRIMARY category
            if not is_primary_category_match(cats, categories):
                dropped_primary_cat += 1
                continue

            # Check year from arXiv ID (more reliable than update_date)
            year = arxiv_id_to_year(paper_id)
            if year is None:
                # Fallback to update_date
                update_date = record.get("update_date", "")
                try:
                    year = int(update_date[:4])
                except (ValueError, TypeError):
                    dropped_year += 1
                    continue
            if year < year_min:
                dropped_year += 1
                continue

            # Check keywords
            if not matches_keywords(title, abstract, keywords):
                dropped_keywords += 1
                continue

            results.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "categories": cats
            })

    print(f"  Filtered: {dropped_year} (year), {dropped_primary_cat} (primary cat), {dropped_keywords} (keywords)")
    return results


def fetch_arxiv_api(config: Dict, limit: int) -> List[Dict]:
    """Fallback: query arXiv API with Atom XML parsing."""
    categories = config["data"]["categories"]
    year_min = config["data"]["year_min"]
    max_papers = config["data"]["max_papers"]
    buffer_limit = int(max_papers * 1.3)
    keywords = config["data"].get("keywords", [])

    # Build search query
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)
    base_url = f"http://export.arxiv.org/api/query?search_query={cat_query}&sortBy=submittedDate&sortOrder=descending"

    results = []
    start = 0
    batch_size = 100
    dropped_year = 0
    dropped_primary_cat = 0
    dropped_keywords = 0

    headers = {"User-Agent": "hallu-rag-student-project/0.1"}

    while len(results) < buffer_limit:
        url = f"{base_url}&start={start}&max_results={batch_size}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            if len(results) >= buffer_limit:
                break

            paper_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()

            # Get categories
            cat_elements = entry.findall("atom:category", ns)
            cats = " ".join(c.get("term") for c in cat_elements)

            # Check PRIMARY category
            if not is_primary_category_match(cats, set(categories)):
                dropped_primary_cat += 1
                continue

            # Check year from published date
            published = entry.find("atom:published", ns).text
            try:
                year = int(published[:4])
            except (ValueError, TypeError):
                dropped_year += 1
                continue
            if year < year_min:
                dropped_year += 1
                continue

            # Check keywords
            if not matches_keywords(title, abstract, keywords):
                dropped_keywords += 1
                continue

            results.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "categories": cats
            })

        start += batch_size
        time.sleep(3)  # arXiv rate limit

    print(f"  Filtered: {dropped_year} (year), {dropped_primary_cat} (primary cat), {dropped_keywords} (keywords)")
    return results


def download_pdf(paper_id: str, raw_dir: Path, failed_log: Path) -> bool:
    """Download a single PDF, return True on success."""
    # Handle old-style IDs with slash
    safe_id = paper_id.replace("/", "_")
    pdf_path = raw_dir / "pdfs" / f"{safe_id}.pdf"

    if pdf_path.exists():
        return True

    url = f"https://arxiv.org/pdf/{paper_id}"
    headers = {"User-Agent": "hallu-rag-student-project/0.1"}

    try:
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        with open(failed_log, "a") as f:
            f.write(f"{paper_id}: {e}\n")
        return False


def print_corpus_report(metadata: List[Dict]):
    """Print corpus statistics."""
    if not metadata:
        print("No papers in corpus.")
        return

    from collections import Counter
    years = []
    primary_cats = []
    for m in metadata:
        year = arxiv_id_to_year(m["paper_id"])
        if year:
            years.append(year)
        cats = m["categories"].strip().split()
        if cats:
            primary_cats.append(cats[0])

    print("\n" + "="*60)
    print("CORPUS REPORT")
    print("="*60)
    print(f"Total papers: {len(metadata)}")
    print(f"Year range: {min(years)} - {max(years)}" if years else "Year range: unknown")
    print("\nPapers per year:")
    for year, count in sorted(Counter(years).items()):
        print(f"  {year}: {count}")
    print("\nPrimary categories:")
    for cat, count in Counter(primary_cats).most_common():
        print(f"  {cat}: {count}")
    print("\nRandom 20 titles:")
    import random
    random.seed(0)
    for m in random.sample(metadata, min(20, len(metadata))):
        print(f"  {m['paper_id']}: {m['title'][:80]}...")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit papers to process (for testing)")
    parser.add_argument("--reprocess", action="store_true", help="Re-extract existing PDFs")
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    max_papers = config["data"]["max_papers"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "pdfs").mkdir(parents=True, exist_ok=True)

    # Step 1: Get metadata
    metadata_path = raw_dir / "metadata_subset.jsonl"
    if metadata_path.exists():
        print("Loading existing metadata...")
        from src.utils import read_jsonl
        metadata = read_jsonl(metadata_path)
    else:
        print("Fetching metadata from Kaggle...")
        metadata = load_kaggle_metadata(config, max_papers)
        if not metadata:
            print("Kaggle file not found, falling back to arXiv API...")
            metadata = fetch_arxiv_api(config, max_papers)

        write_jsonl(metadata_path, metadata)
        print(f"Saved {len(metadata)} metadata records to {metadata_path}")

    # Print corpus report
    print_corpus_report(metadata)

    # Step 2: Download PDFs
    print("Downloading PDFs...")
    failed_log = raw_dir / "failed.txt"
    success_count = 0

    for meta in tqdm(metadata, desc="Downloading PDFs"):
        if success_count >= max_papers:
            break

        safe_id = meta["paper_id"].replace("/", "_")
        already_downloaded = (raw_dir / "pdfs" / f"{safe_id}.pdf").exists()
        if download_pdf(meta["paper_id"], raw_dir, failed_log):
            success_count += 1
        if not already_downloaded:
            time.sleep(3)  # Rate limit

    print(f"Successfully downloaded {success_count} PDFs")

    # Step 3: Extract and process papers
    print("Extracting and cleaning papers...")
    processed_count = 0

    for meta in tqdm(metadata, desc="Processing papers"):
        if processed_count >= max_papers:
            break

        safe_id = meta["paper_id"].replace("/", "_")
        pdf_path = raw_dir / "pdfs" / f"{safe_id}.pdf"
        output_path = processed_dir / f"{safe_id}.json"

        if not pdf_path.exists():
            continue

        if output_path.exists() and not args.reprocess:
            processed_count += 1
            continue

        paper = extract_paper(str(pdf_path), meta)
        if paper:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(paper, f, ensure_ascii=False, indent=2)
            processed_count += 1

    print(f"Processed {processed_count} papers to {processed_dir}")


if __name__ == "__main__":
    main()