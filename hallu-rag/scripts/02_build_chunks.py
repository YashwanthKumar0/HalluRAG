#!/usr/bin/env python
"""
Script 2: Build chunks from processed papers.
Run after 01_download_data.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.chunking import run_chunking
from src.config import get_config
from src.utils import setup_logging


def main():
    logger = setup_logging()
    logger.info("Starting chunking...")

    cfg = get_config()

    # Run chunking for both strategies
    results = run_chunking()

    for strategy, chunks in results.items():
        logger.info(f"{strategy}: {len(chunks)} chunks created")

    logger.info(f"Chunks saved to: {cfg.data.get('chunks_dir')}")


if __name__ == "__main__":
    main()