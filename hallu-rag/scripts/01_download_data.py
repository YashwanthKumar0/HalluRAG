#!/usr/bin/env python
"""
Script 1: Download arXiv papers and extract text.
Run this first to prepare the data.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_prep import run_data_prep
from src.config import get_config
from src.utils import setup_logging


def main():
    logger = setup_logging()
    logger.info("Starting data download and processing...")

    cfg = get_config()

    # Run data preparation
    papers = run_data_prep()

    logger.info(f"Completed! Processed {len(papers)} papers.")
    logger.info(f"Papers saved to: {cfg.data.get('processed_dir')}")


if __name__ == "__main__":
    main()