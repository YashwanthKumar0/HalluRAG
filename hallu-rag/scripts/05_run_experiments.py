#!/usr/bin/env python
"""
Script 5: Run all experiments and save results.
Run after 04_draft_gold_set.py (and after filling in the gold Q&A)
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evaluation import run_all_experiments, save_results
from src.indexing import load_indexes
from src.config import get_config
from src.utils import setup_logging


def main():
    logger = setup_logging()
    logger.info("Running all experiments...")

    cfg = get_config()

    # Load indexes
    logger.info("Loading indexes...")
    indexes = load_indexes()

    if not indexes:
        logger.error("No indexes found! Run 03_build_indexes.py first.")
        sys.exit(1)

    # Run experiments
    results = run_all_experiments(indexes=indexes)

    # Save results
    output_dir = Path(cfg.data.get("index_dir", "data/index")).parent / "results"
    save_results(results, output_dir)

    # Print summary
    logger.info("\n=== EXPERIMENT SUMMARY ===")
    for result in results:
        config = result.config
        metrics = result.metrics
        config_str = f"{config['strategy']}/{config['retrieval_type']}/rerank={config['use_rerank']}"
        logger.info(f"\n{config_str}:")
        logger.info(f"  Recall@1: {metrics.get('recall@1', 0):.3f}")
        logger.info(f"  Recall@5: {metrics.get('recall@5', 0):.3f}")
        logger.info(f"  MRR: {metrics.get('mrr', 0):.3f}")
        logger.info(f"  Faithfulness: {metrics.get('faithfulness_mean', 0):.3f}")
        logger.info(f"  Relevance (ROUGE-L): {metrics.get('relevance_rougeL_f', 0):.3f}")
        if 'cohen_kappa' in metrics:
            logger.info(f"  Cohen's Kappa: {metrics['cohen_kappa']:.3f}")

    logger.info(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()