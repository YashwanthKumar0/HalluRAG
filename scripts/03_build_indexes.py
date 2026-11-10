#!/usr/bin/env python
"""
Build FAISS and BM25 indexes for both chunking strategies.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.indexing import build_all_indexes


def main():
    config = load_config()
    build_all_indexes(config)


if __name__ == "__main__":
    main()