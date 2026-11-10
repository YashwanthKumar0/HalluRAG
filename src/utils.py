import json
from typing import List, Dict, Any


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """Write list of dicts to a JSONL file (overwrites)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    """Append a single dict to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text))