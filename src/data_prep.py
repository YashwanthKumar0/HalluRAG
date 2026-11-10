import re
import fitz
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


def extract_paper(pdf_path: str, meta: Dict) -> Optional[Dict]:
    """Extract and clean text from a PDF paper using PyMuPDF."""
    doc = fitz.open(pdf_path)

    # Extract text page by page with font info
    pages_data = []
    for page in doc:
        # Get text with font info
        blocks = page.get_text("dict")["blocks"]
        page_text = page.get_text("text")
        pages_data.append({"blocks": blocks, "text": page_text})

    doc.close()

    if not pages_data:
        return None

    # Remove headers/footers: lines that repeat on >= 50% of pages
    all_lines = []
    for page_data in pages_data:
        lines = [line.strip() for line in page_data["text"].split('\n') if line.strip()]
        all_lines.append(lines)

    line_counts = Counter()
    for lines in all_lines:
        unique_lines = set(lines)
        for line in unique_lines:
            line_counts[line] += 1

    threshold = len(pages_data) * 0.5
    repeated_lines = {line for line, count in line_counts.items() if count >= threshold}

    # Also remove page-number-only lines
    page_number_pattern = re.compile(r'^\d+$')

    cleaned_pages = []
    for page_data in pages_data:
        lines = [line.strip() for line in page_data["text"].split('\n') if line.strip()]
        cleaned = []
        for line in lines:
            if line in repeated_lines:
                continue
            if page_number_pattern.match(line):
                continue
            cleaned.append(line)
        cleaned_pages.append(cleaned)

    # Join all lines back
    full_text = '\n'.join('\n'.join(page_lines) for page_lines in cleaned_pages)

    # Fix hyphenation at line ends
    full_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', full_text)

    # Collapse whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = full_text.strip()

    # Discard the title page and publication metadata before the abstract.
    abstract_match = re.search(r'\babstract\b', full_text, re.IGNORECASE)
    if abstract_match:
        full_text = full_text[abstract_match.start():].strip()

    # Cut references at last "References" or "Bibliography" heading
    ref_pattern = re.compile(r'^(references|bibliography)\s*$', re.IGNORECASE | re.MULTILINE)
    matches = list(ref_pattern.finditer(full_text))
    if matches:
        last_match = matches[-1]
        full_text = full_text[:last_match.start()].strip()

    # Drop figure/table captions
    caption_pattern = re.compile(
        r'(Figure|Fig\.|Table)\s+\d+[.:]?\s*.*?(?=\n|$)',
        re.IGNORECASE
    )
    full_text = caption_pattern.sub('', full_text)

    # Collapse whitespace again after caption removal
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # Skip if too short
    if len(full_text) < 2000:
        return None

    # Detect sections using font size heuristic
    sections = detect_sections(pages_data)
    abstract_index = next(
        (i for i, section in enumerate(sections)
         if re.fullmatch(r"(?:\d+[\.\s]*)?abstract\.?", section["heading"].strip(),
                         re.IGNORECASE)),
        None,
    )
    if abstract_index is not None:
        sections = sections[abstract_index:]
    else:
        sections = [{"heading": "Abstract", "text": full_text}]

    return {
        "paper_id": meta.get("paper_id", ""),
        "title": meta.get("title", ""),
        "abstract": meta.get("abstract", ""),
        "categories": meta.get("categories", ""),
        "sections": sections
    }


def detect_sections(pages_data: List[Dict]) -> List[Dict[str, str]]:
    """
    Detect sections using font size heuristic from PyMuPDF.
    A line is a heading if its max span font size >= 1.15 x page's median body font size,
    it is < 80 chars, and has no ending period.
    """
    # First, compute median body font size per page
    page_median_sizes = []
    for page_data in pages_data:
        sizes = []
        for block in page_data["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes.append(span["size"])
        if sizes:
            page_median_sizes.append(np.median(sizes))
        else:
            page_median_sizes.append(10.0)  # fallback

    overall_median = np.median(page_median_sizes) if page_median_sizes else 10.0
    heading_threshold = overall_median * 1.15

    # Now extract lines with font info and detect headings
    heading_keywords = (
        r'(Abstract|Introduction|Related Work|Background|Method|Methods|Methodology|'
        r'Approach|Experiment|Experiments|Results|Discussion|Conclusion|Conclusions|'
        r'Limitation|Limitations|Future Work|Acknowledgments?|References|Bibliography)'
    )
    heading_pattern = re.compile(
        rf'^(\d+(\.\d+)*)?\s*({heading_keywords})\.?$',
        re.IGNORECASE
    )

    sections = []
    current_heading = "Body"
    current_text = []

    for page_idx, page_data in enumerate(pages_data):
        page_median = page_median_sizes[page_idx] if page_idx < len(page_median_sizes) else overall_median
        page_threshold = page_median * 1.15

        for block in page_data["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                # Get max font size in this line
                max_size = max((span["size"] for span in line["spans"]), default=0)
                line_text = "".join(span["text"] for span in line["spans"]).strip()

                if not line_text:
                    continue

                # Check if heading: large font, short, no period, matches pattern
                is_heading = (
                    max_size >= page_threshold and
                    len(line_text) < 80 and
                    not line_text.endswith('.') and
                    (heading_pattern.match(line_text) or
                     (len(line_text) < 50 and max_size >= page_threshold * 1.1))
                )

                if is_heading:
                    # Save previous section
                    if current_text:
                        sections.append({
                            "heading": current_heading,
                            "text": ' '.join(current_text).strip()
                        })
                    current_heading = line_text
                    current_text = []
                else:
                    current_text.append(line_text)

    # Don't forget the last section
    if current_text:
        sections.append({
            "heading": current_heading,
            "text": ' '.join(current_text).strip()
        })

    # If no sections found, return single Body section
    if not sections:
        # Fallback: use simple text-based detection
        full_text = ' '.join(page_data["text"] for page_data in pages_data)
        sections = detect_sections_simple(full_text)

    return sections


def detect_sections_simple(text: str) -> List[Dict[str, str]]:
    """Fallback section detection using regex on plain text."""
    heading_keywords = (
        r'(Abstract|Introduction|Related Work|Background|Method|Methods|Methodology|'
        r'Approach|Experiment|Experiments|Results|Discussion|Conclusion|Conclusions|'
        r'Limitation|Limitations|Future Work|Acknowledgments?|References|Bibliography)'
    )
    heading_pattern = re.compile(
        rf'^(\d+(\.\d+)*)?\s*({heading_keywords})\.?$',
        re.IGNORECASE
    )

    lines = text.split('\n')
    sections = []
    current_heading = "Body"
    current_text = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if len(stripped) < 80 and heading_pattern.match(stripped):
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "text": ' '.join(current_text).strip()
                })
            current_heading = stripped
            current_text = []
        else:
            current_text.append(stripped)

    if current_text:
        sections.append({
            "heading": current_heading,
            "text": ' '.join(current_text).strip()
        })

    if not sections:
        sections = [{"heading": "Body", "text": text}]

    return sections


def filter_chunks(chunks: List[Dict], min_words: int = 40, min_alpha_ratio: float = 0.6) -> Tuple[List[Dict], int]:
    """Filter out low-quality chunks: too few words or too few alphabetic tokens."""
    filtered = []
    dropped_count = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        words = text.split()
        if len(words) < min_words:
            dropped_count += 1
            continue
        # Count alphabetic tokens
        alpha_tokens = sum(1 for w in words if any(c.isalpha() for c in w))
        if len(words) > 0 and (alpha_tokens / len(words)) < min_alpha_ratio:
            dropped_count += 1
            continue
        filtered.append(chunk)
    return filtered, dropped_count


# Need numpy import
import numpy as np