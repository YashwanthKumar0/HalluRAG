#!/usr/bin/env python
"""
Script 7: Compute human agreement metrics.
Run after human labels are added to data/gold/human_labels.jsonl
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.evaluation import compute_human_agreement
from src.config import get_config
from src.utils import setup_logging, load_jsonl


def main():
    logger = setup_logging()
    logger.info("Computing human agreement...")

    cfg = get_config()

    # Load human labels
    human_labels_path = Path(cfg.evaluation.get("human_labels_path", "data/gold/human_labels.jsonl"))
    if not human_labels_path.exists():
        logger.error(f"Human labels not found at {human_labels_path}")
        logger.info("Please run 06_export_for_human_labels.py first, then fill in labels.")
        sys.exit(1)

    human_labels = load_jsonl(human_labels_path)
    logger.info(f"Loaded {len(human_labels)} human labels")

    # Load predictions
    results_dir = Path(cfg.data.get("index_dir", "data/index")).parent / "results"
    predictions_path = results_dir / "predictions_for_labeling.jsonl"
    if not predictions_path.exists():
        logger.error(f"Predictions not found at {predictions_path}")
        logger.info("Please run 06_export_for_human_labels.py first.")
        sys.exit(1)

    predictions = load_jsonl(predictions_path)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Group by config
    configs = {}
    for p in predictions:
        config = p["config"]
        if config not in configs:
            configs[config] = {"predictions": [], "human_labels": {}}
        configs[config]["predictions"].append(p)

    # Map human labels by question_id
    human_label_map = {hl["question_id"]: hl["faithfulness_label"] for hl in human_labels}

    # Compute agreement for each config
    logger.info("\n=== HUMAN AGREEMENT RESULTS ===\n")

    for config, data in configs.items():
        preds = data["predictions"]

        # Match predictions with human labels
        matched_predictions = []
        matched_human = []

        for p in preds:
            qid = p["question_id"]
            if qid in human_label_map:
                matched_predictions.append(p["faithfulness_label"])
                matched_human.append(human_label_map[qid])

        if not matched_predictions:
            logger.warning(f"No matching labels for config {config}")
            continue

        agreement = compute_human_agreement(matched_predictions, matched_human)

        logger.info(f"Config: {config}")
        logger.info(f"  Questions matched: {len(matched_predictions)}")
        logger.info(f"  Cohen's Kappa: {agreement['cohen_kappa']:.4f}")
        logger.info(f"  Accuracy: {agreement['accuracy']:.4f}")
        for key, val in agreement.items():
            if key not in ["cohen_kappa", "accuracy"]:
                logger.info(f"  {key}: {val:.4f}")
        logger.info("")

    # Overall agreement (across all configs)
    all_predictions = []
    all_human = []
    for p in predictions:
        qid = p["question_id"]
        if qid in human_label_map:
            all_predictions.append(p["faithfulness_label"])
            all_human.append(human_label_map[qid])

    if all_predictions:
        overall = compute_human_agreement(all_predictions, all_human)
        logger.info("=== OVERALL (all configs) ===")
        logger.info(f"  Questions matched: {len(all_predictions)}")
        logger.info(f"  Cohen's Kappa: {overall['cohen_kappa']:.4f}")
        logger.info(f"  Accuracy: {overall['accuracy']:.4f}")


if __name__ == "__main__":
    main()