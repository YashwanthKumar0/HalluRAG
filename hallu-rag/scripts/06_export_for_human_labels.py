#!/usr/bin/env python
"""
Script 6: Export predictions for human labeling.
Run after 05_run_experiments.py to get system predictions for human evaluation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evaluation import run_all_experiments
from src.indexing import load_indexes
from src.config import get_config
from src.utils import setup_logging, save_jsonl, ensure_dir


def main():
    logger = setup_logging()
    logger.info("Exporting predictions for human labeling...")

    cfg = get_config()

    # Load indexes
    logger.info("Loading indexes...")
    indexes = load_indexes()

    if not indexes:
        logger.error("No indexes found! Run 03_build_indexes.py first.")
        sys.exit(1)

    # Load gold Q&A
    gold_qa_path = Path(cfg.evaluation.get("gold_qa_path", "data/gold/gold_qa.jsonl"))
    from src.utils import load_jsonl
    gold_qa = load_jsonl(gold_qa_path)

    # Run experiments and collect predictions
    all_predictions = []

    strategies = cfg.evaluation.get("experiments", {}).get("chunking_strategies", ["fixed", "semantic"])
    retrieval_types = cfg.evaluation.get("experiments", {}).get("retrieval_types", ["dense", "hybrid"])
    rerank_options = cfg.evaluation.get("experiments", {}).get("rerank_options", [False, True])

    for strategy in strategies:
        for retrieval_type in retrieval_types:
            for use_rerank in rerank_options:
                config_str = f"{strategy}_{retrieval_type}_rerank{use_rerank}"
                logger.info(f"Running {config_str} for predictions...")

                from src.pipeline import create_pipeline
                pipeline = create_pipeline(
                    strategy=strategy,
                    retrieval_type=retrieval_type,
                    use_rerank=use_rerank,
                    indexes=indexes,
                )

                for qa in gold_qa:
                    question = qa["question"]
                    question_id = qa.get("question_id", "")

                    result = pipeline.run(question)

                    all_predictions.append({
                        "question_id": question_id,
                        "question": question,
                        "config": config_str,
                        "strategy": strategy,
                        "retrieval_type": retrieval_type,
                        "use_rerank": use_rerank,
                        "answer": result.answer,
                        "faithfulness_label": result.verification["overall_faithfulness"],
                        "faithfulness_score": result.verification["faithfulness_score"],
                        "supported_count": result.verification["supported_count"],
                        "unverified_count": result.verification["unverified_count"],
                        "unsupported_count": result.verification["unsupported_count"],
                        "claims": result.verification["claims"],
                    })

    # Save predictions
    output_dir = Path(cfg.data.get("index_dir", "data/index")).parent / "results"
    ensure_dir(output_dir)
    output_path = output_dir / "predictions_for_labeling.jsonl"
    save_jsonl(all_predictions, output_path)

    logger.info(f"Saved {len(all_predictions)} predictions to {output_path}")
    logger.info("Now you can:")
    logger.info("  1. Review the predictions")
    logger.info("  2. Manually label faithfulness for each answer")
    logger.info("  3. Save labels to data/gold/human_labels.jsonl")
    logger.info("  4. Run 07_human_agreement.py")


if __name__ == "__main__":
    main()