#!/usr/bin/env python
"""
Script 3: Build FAISS and BM25 indexes.
Run after 02_build_chunks.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.indexing import build_all_indexes
from src.config import get_config
from src.utils import setup_logging


def main():
    logger = setup_logging()
    logger.info("Starting index building...")

    cfg = get_config()

    # Build indexes for both strategies
    results = build_all_indexes()

    for strategy, indexes in results.items():
        logger.info(f"{strategy}: {indexes['num_chunks']} chunks indexed")

    logger.info(f"Indexes saved to: {cfg.data.get('index_dir')}")


if __name__ == "__main__":
    main()