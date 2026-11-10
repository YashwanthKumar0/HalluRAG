"""
Utility functions for Hallucination-Aware RAG.
"""
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level

    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def generate_chunk_id(text: str, prefix: str = "C") -> str:
    """
    Generate a deterministic chunk ID from text content.

    Args:
        text: Text content
        prefix: Prefix for the ID

    Returns:
        Unique chunk ID
    """
    hash_obj = hashlib.md5(text.encode("utf-8"))
    return f"{prefix}{hash_obj.hexdigest()[:8]}"


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save list of dictionaries as JSONL.

    Args:
        data: List of dictionaries to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file.

    Args:
        path: Input file path

    Returns:
        List of dictionaries
    """
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data as JSON.

    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    return sent_tokenize(text)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    import re

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Compute hash of a file.

    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha256)

    Returns:
        Hex digest of file hash
    """
    path = Path(path)
    hash_func = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


if __name__ == "__main__":
    # Smoke test
    logger = setup_logging()
    logger.info("Logging works")

    test_text = "This is a test. This is another sentence! And a third one?"
    sentences = split_sentences(test_text)
    print(f"Sentences: {sentences}")

    chunk_id = generate_chunk_id("test content")
    print(f"Chunk ID: {chunk_id}")

    test_data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    save_jsonl(test_data, "test_output.jsonl")
    loaded = load_jsonl("test_output.jsonl")
    print(f"Loaded: {loaded}")

    print("All utils tests passed!")