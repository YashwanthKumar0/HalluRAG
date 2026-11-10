from functools import lru_cache
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer
import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_name: str):
    """Load and cache HF tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer_name)

def fixed_chunks(paper: Dict, chunk_tokens: int = 512, overlap_tokens: int = 50, tokenizer_name: str = "BAAI/bge-small-en-v1.5") -> List[Dict]:
    """
    Tokenize each section's text, slide window of chunk_tokens with stride chunk_tokens - overlap_tokens.
    Chunks do not cross section boundaries.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    stride = chunk_tokens - overlap_tokens
    chunks = []
    chunk_idx = 0

    for section in paper.get("sections", []):
        section_name = section.get("heading", "Body")
        text = section.get("text", "")

        if not text.strip():
            continue

        # Tokenize the section text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Slide window
        for start in range(0, len(tokens), stride):
            end = min(start + chunk_tokens, len(tokens))
            chunk_tokens_slice = tokens[start:end]

            if len(chunk_tokens_slice) == 0:
                break

            # Decode back to text
            chunk_text = tokenizer.decode(chunk_tokens_slice, skip_special_tokens=True)
            n_tokens = len(chunk_tokens_slice)

            chunks.append({
                "chunk_id": f"{paper['paper_id']}_fixed_{chunk_idx:04d}",
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "section": section_name,
                "text": chunk_text,
                "n_tokens": n_tokens,
                "strategy": "fixed"
            })
            chunk_idx += 1

            if end >= len(tokens):
                break

    return chunks


def semantic_chunks(paper: Dict, max_tokens: int = 512, min_tokens: int = 80, tokenizer_name: str = "BAAI/bge-small-en-v1.5") -> List[Dict]:
    """
    Split sections into paragraphs, then sentences. Greedily merge until max_tokens.
    Merge any chunk < min_tokens into neighbour. Never cut mid-sentence.
    """
    tokenizer = get_tokenizer(tokenizer_name)
    chunks = []
    chunk_idx = 0

    for section in paper.get("sections", []):
        section_name = section.get("heading", "Body")
        text = section.get("text", "")

        if not text.strip():
            continue

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Process each paragraph
        units = []  # list of (text, token_count)
        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
            if para_tokens <= max_tokens:
                units.append((para, para_tokens))
            else:
                # Split long paragraph into sentences
                sentences = nltk.sent_tokenize(para)
                for sent in sentences:
                    sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                    if sent_tokens <= max_tokens:
                        units.append((sent, sent_tokens))
                    else:
                         # Handle exceptionally long sentences
                         encoded = tokenizer.encode(sent, add_special_tokens=False)
                         truncated = tokenizer.decode(
                                encoded[:max_tokens],
                                skip_special_tokens=True
                         )
                         units.append((truncated, min(len(encoded), max_tokens)))
        # Greedily merge units into chunks
        current_chunk_parts = []
        current_tokens = 0

        for unit_text, unit_tokens in units:
            if current_tokens + unit_tokens > max_tokens and current_chunk_parts:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_parts)
                n_tokens = current_tokens
                chunks.append({
                    "chunk_id": f"{paper['paper_id']}_semantic_{chunk_idx:04d}",
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "section": section_name,
                    "text": chunk_text,
                    "n_tokens": n_tokens,
                    "strategy": "semantic"
                })
                chunk_idx += 1
                current_chunk_parts = [unit_text]
                current_tokens = unit_tokens
            else:
                current_chunk_parts.append(unit_text)
                current_tokens += unit_tokens

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts)
            n_tokens = current_tokens
            chunks.append({
                "chunk_id": f"{paper['paper_id']}_semantic_{chunk_idx:04d}",
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "section": section_name,
                "text": chunk_text,
                "n_tokens": n_tokens,
                "strategy": "semantic"
            })
            chunk_idx += 1

        # Post-process: merge small chunks without crossing section boundaries
    merged_chunks = []
    i = 0

    while i < len(chunks):
        current = chunks[i]

        if current["n_tokens"] < min_tokens:
            # Prefer merging within a section, but avoid a run of undersized
            # chunks when a section is shorter than min_tokens.
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]

                same_section = (
                    current["paper_id"] == next_chunk["paper_id"]
                    and current["section"] == next_chunk["section"]
                )

                if same_section or next_chunk["n_tokens"] < min_tokens:
                    merged_text = current["text"] + " " + next_chunk["text"]
                    merged_tokens = len(
                        tokenizer.encode(
                            merged_text,
                            add_special_tokens=False
                        )
                    )

                    if merged_tokens <= max_tokens:
                        merged_chunks.append({
                            **current,
                            "text": merged_text,
                            "n_tokens": merged_tokens
                        })
                        i += 2
                        continue

            # Otherwise, try merging with the previous chunk
            if merged_chunks:
                previous = merged_chunks[-1]

                same_section = (
                    current["paper_id"] == previous["paper_id"]
                    and current["section"] == previous["section"]
                )

                if same_section or current["n_tokens"] < min_tokens:
                    merged_text = previous["text"] + " " + current["text"]
                    merged_tokens = len(
                        tokenizer.encode(
                            merged_text,
                            add_special_tokens=False
                        )
                    )

                    if merged_tokens <= max_tokens:
                        merged_chunks[-1]["text"] = merged_text
                        merged_chunks[-1]["n_tokens"] = merged_tokens
                        i += 1
                        continue

            # Keep small chunks if safe merging is not possible
            merged_chunks.append(current)
        else:
            merged_chunks.append(current)

        i += 1

    return merged_chunks