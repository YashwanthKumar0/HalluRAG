"""
Data preparation module for downloading and processing arXiv papers.
Phase 1: Download PDFs, extract text, and save as clean JSON.
"""
import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import arxiv
import fitz  # PyMuPDF
from tqdm import tqdm

from .config import get_config
from .utils import setup_logging, save_json, clean_text, ensure_dir


logger = setup_logging()


@dataclass
class Paper:
    """Represents an arXiv paper with metadata and extracted text."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    updated: str
    pdf_url: str
    text: str = ""
    page_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def download_papers(
    categories: List[str],
    max_papers_per_category: int,
    date_from: str,
    date_to: str,
    output_dir: Path,
) -> List[Paper]:
    """
    Download papers from arXiv for given categories.

    Args:
        categories: List of arXiv categories (e.g., ["cs.CL", "cs.AI"])
        max_papers_per_category: Maximum papers per category
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        output_dir: Directory to save PDFs

    Returns:
        List of Paper objects
    """
    ensure_dir(output_dir)

    # Build search query
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    date_query = f"submittedDate:[{date_from} TO {date_to}]"
    query = f"({cat_query}) AND {date_query}"

    logger.info(f"Searching arXiv with query: {query}")

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be nice to arXiv API
        num_retries=3,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_papers_per_category * len(categories),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in tqdm(client.results(search), desc="Fetching paper metadata"):
        paper = Paper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title,
            authors=[str(a) for a in result.authors],
            abstract=result.summary,
            categories=result.categories,
            published=result.published.strftime("%Y-%m-%d"),
            updated=result.updated.strftime("%Y-%m-%d"),
            pdf_url=result.pdf_url,
        )
        papers.append(paper)

        # Download PDF
        pdf_path = output_dir / f"{paper.arxiv_id}.pdf"
        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(output_dir), filename=f"{paper.arxiv_id}.pdf")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to download PDF for {paper.arxiv_id}: {e}")

    logger.info(f"Downloaded metadata for {len(papers)} papers")
    return papers


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (extracted_text, page_count)
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
        doc.close()
        full_text = "\n".join(text_parts)
        return clean_text(full_text), len(doc)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return "", 0


def clean_paper_text(text: str) -> str:
    """
    Clean extracted paper text by removing headers, footers, references, etc.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove page numbers and headers/footers (simple heuristic)
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip very short lines that are likely page numbers/headers
        if len(line) < 3:
            continue
        # Skip lines that are just numbers (page numbers)
        if line.isdigit():
            continue
        # Skip common header/footer patterns
        if re.match(r"^(arXiv:|doi:|ISBN|©|Proceedings of|Conference on)", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove reference section (simple heuristic - everything after "References" or "Bibliography")
    ref_patterns = [
        r"\n\s*References\s*\n",
        r"\n\s*Bibliography\s*\n",
        r"\n\s*REFERENCES\s*\n",
        r"\n\s*BIBLIOGRAPHY\s*\n",
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[: match.start()]
            break

    return clean_text(text)


def process_papers(papers: List[Paper], raw_dir: Path, processed_dir: Path) -> List[Paper]:
    """
    Process downloaded papers: extract text from PDFs and save as JSON.

    Args:
        papers: List of Paper objects
        raw_dir: Directory with PDFs
        processed_dir: Output directory for processed JSON

    Returns:
        List of processed Paper objects with text
    """
    ensure_dir(processed_dir)

    processed_papers = []

    for paper in tqdm(papers, desc="Processing papers"):
        pdf_path = raw_dir / f"{paper.arxiv_id}.pdf"

        if pdf_path.exists():
            text, page_count = extract_text_from_pdf(pdf_path)
            paper.text = clean_paper_text(text)
            paper.page_count = page_count
        else:
            logger.warning(f"PDF not found for {paper.arxiv_id}, using abstract only")
            paper.text = paper.abstract
            paper.page_count = 0

        # Save individual paper JSON
        output_path = processed_dir / f"{paper.arxiv_id}.json"
        save_json(paper.to_dict(), output_path)

        processed_papers.append(paper)

    logger.info(f"Processed {len(processed_papers)} papers")
    return processed_papers


def run_data_prep(
    categories: Optional[List[str]] = None,
    max_papers_per_category: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> List[Paper]:
    """
    Run the full data preparation pipeline.

    Args:
        categories: arXiv categories (from config if None)
        max_papers_per_category: Max papers per category (from config if None)
        date_from: Start date (from config if None)
        date_to: End date (from config if None)
        raw_dir: Raw PDF directory (from config if None)
        processed_dir: Processed JSON directory (from config if None)

    Returns:
        List of processed Paper objects
    """
    cfg = get_config()

    categories = categories or cfg.data.get("categories", ["cs.CL", "cs.AI"])
    max_papers = max_papers_per_category or cfg.data.get("max_papers_per_category", 100)
    date_from = date_from or cfg.data.get("date_from", "2023-01-01")
    date_to = date_to or cfg.data.get("date_to", "2024-12-31")
    raw_dir = Path(raw_dir or cfg.data.get("raw_dir", "data/raw"))
    processed_dir = Path(processed_dir or cfg.data.get("processed_dir", "data/processed"))

    logger.info("Starting data preparation pipeline")

    # Download papers
    papers = download_papers(categories, max_papers, date_from, date_to, raw_dir)

    # Process papers
    processed_papers = process_papers(papers, raw_dir, processed_dir)

    logger.info("Data preparation complete")
    return processed_papers


if __name__ == "__main__":
    # Smoke test: just verify imports work and config loads
    cfg = get_config()
    print(f"Config loaded: {cfg.data.get('categories')}")
    print("data_prep module loaded successfully")