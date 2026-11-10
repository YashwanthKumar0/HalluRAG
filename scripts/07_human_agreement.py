#!/usr/bin/env python
"""
Compute human agreement metrics for NLI label validation.

Reads human_labels_todo.csv with columns:
- nli_label (model's label)
- human_label_a, human_label_b (two annotators)
- human_label (agreed final label)

Computes:
- % agreement and Cohen's κ between NLI label and final human label (3-class and binary)
- Inter-annotator κ between the two humans
- Confusion matrix
- Writes results/human_agreement.md; prints PASS if binary agreement ≥ 80%.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score


def load_labels(csv_path: str) -> pd.DataFrame:
    """Load and validate human labels CSV."""
    df = pd.read_csv(csv_path)
    required = ["nli_label", "human_label_a", "human_label_b", "human_label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def normalize_label(label: str) -> str:
    """Normalize label to standard format."""
    if pd.isna(label):
        return ""
    label = str(label).strip().lower()
    if label in ["supported", "support"]:
        return "supported"
    elif label in ["unsupported", "unsupport", "unverified", "unverified"]:
        return "unsupported"
    elif label in ["unverified", "verify", "unverified"]:
        return "unverified"
    return label


def to_binary(label: str) -> int:
    """Convert 3-class to binary: supported=1, else=0."""
    label = normalize_label(label)
    return 1 if label == "supported" else 0


def compute_kappa(y1: List[str], y2: List[str], labels: List[str] = None) -> float:
    """Compute Cohen's kappa."""
    # Normalize
    y1_norm = [normalize_label(l) for l in y1]
    y2_norm = [normalize_label(l) for l in y2]

    if labels is None:
        labels = sorted(set(y1_norm + y2_norm))

    # Map to indices
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y1_idx = [label_to_idx.get(l, -1) for l in y1_norm]
    y2_idx = [label_to_idx.get(l, -1) for l in y2_norm]

    # Filter valid
    valid = [(a, b) for a, b in zip(y1_idx, y2_idx) if a != -1 and b != -1]
    if not valid:
        return 0.0

    y1_valid, y2_valid = zip(*valid)
    return cohen_kappa_score(y1_valid, y2_valid, labels=range(len(labels)))


def compute_agreement(df: pd.DataFrame) -> Dict:
    """Compute all agreement metrics."""
    results = {}

    # Normalize all labels
    for col in ["nli_label", "human_label_a", "human_label_b", "human_label"]:
        df[col] = df[col].apply(normalize_label)

    labels = ["supported", "unsupported", "unverified"]

    # Check if human labels are filled (not empty/na)
    has_human_labels = df["human_label"].notna().any() and (df["human_label"] != "").any()
    has_annotator_labels = (df["human_label_a"].notna().any() and (df["human_label_a"] != "").any() and
                            df["human_label_b"].notna().any() and (df["human_label_b"] != "").any())

    # 1. NLI vs Final Human (3-class)
    if has_human_labels:
        results["nli_vs_human_3class"] = {
            "accuracy": accuracy_score(df["human_label"], df["nli_label"]),
            "kappa": compute_kappa(df["nli_label"].tolist(), df["human_label"].tolist(), labels)
        }

        # 2. NLI vs Final Human (binary)
        nli_binary = df["nli_label"].apply(to_binary)
        human_binary = df["human_label"].apply(to_binary)
        results["nli_vs_human_binary"] = {
            "accuracy": accuracy_score(human_binary, nli_binary),
            "kappa": cohen_kappa_score(human_binary, nli_binary)
        }

        # 5. Confusion matrix (NLI vs Human, 3-class)
        # Only include labels present in the data
        present_labels = [l for l in labels if l in df["human_label"].values or l in df["nli_label"].values]
        cm = confusion_matrix(
            df["human_label"], df["nli_label"],
            labels=present_labels
        )
        results["confusion_matrix"] = {
            "labels": present_labels,
            "matrix": cm.tolist()
        }

        # Per-class metrics
        results["per_class"] = {}
        for i, label in enumerate(present_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results["per_class"][label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[i, :].sum())
            }
    else:
        results["nli_vs_human_3class"] = {"accuracy": None, "kappa": None}
        results["nli_vs_human_binary"] = {"accuracy": None, "kappa": None}
        results["confusion_matrix"] = {"labels": labels, "matrix": []}
        results["per_class"] = {}

    # 3. Inter-annotator (3-class)
    if has_annotator_labels:
        results["annotator_3class"] = {
            "accuracy": accuracy_score(df["human_label_a"], df["human_label_b"]),
            "kappa": compute_kappa(df["human_label_a"].tolist(), df["human_label_b"].tolist(), labels)
        }

        # 4. Inter-annotator (binary)
        a_binary = df["human_label_a"].apply(to_binary)
        b_binary = df["human_label_b"].apply(to_binary)
        results["annotator_binary"] = {
            "accuracy": accuracy_score(a_binary, b_binary),
            "kappa": cohen_kappa_score(a_binary, b_binary)
        }
    else:
        results["annotator_3class"] = {"accuracy": None, "kappa": None}
        results["annotator_binary"] = {"accuracy": None, "kappa": None}

    return results


def write_report(results: Dict, output_path: str, df: pd.DataFrame):
    """Write agreement report to markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Human-NLI Agreement Report\n\n")

        # Summary
        f.write("## Summary\n\n")

        # Handle None values
        binary_acc = results.get("nli_vs_human_binary", {}).get("accuracy")
        if binary_acc is not None:
            f.write(f"- **Binary Agreement (NLI vs Human)**: {binary_acc:.2%}")
            f.write(f" {'✅ PASS' if binary_acc >= 0.80 else '❌ FAIL'} (≥80% required)\n")
        else:
            f.write("- **Binary Agreement (NLI vs Human)**: N/A (human labels not filled)\n")

        acc_3class = results.get("nli_vs_human_3class", {}).get("accuracy")
        if acc_3class is not None:
            f.write(f"- **3-Class Agreement (NLI vs Human)**: {acc_3class:.2%}\n")
        else:
            f.write("- **3-Class Agreement (NLI vs Human)**: N/A (human labels not filled)\n")

        inter_bin = results.get("annotator_binary", {}).get("accuracy")
        if inter_bin is not None:
            f.write(f"- **Inter-Annotator Binary Agreement**: {inter_bin:.2%}\n")
        else:
            f.write("- **Inter-Annotator Binary Agreement**: N/A (annotator labels not filled)\n")

        inter_3class = results.get("annotator_3class", {}).get("accuracy")
        if inter_3class is not None:
            f.write(f"- **Inter-Annotator 3-Class Agreement**: {inter_3class:.2%}\n\n")
        else:
            f.write("- **Inter-Annotator 3-Class Agreement**: N/A (annotator labels not filled)\n\n")

        # Kappa scores
        f.write("## Cohen's Kappa\n\n")
        f.write("| Comparison | 3-Class κ | Binary κ |\n")
        f.write("|------------|-----------|----------|\n")

        kappa_3 = results.get("nli_vs_human_3class", {}).get("kappa")
        kappa_bin = results.get("nli_vs_human_binary", {}).get("kappa")
        kappa_3_annot = results.get("annotator_3class", {}).get("kappa")
        kappa_bin_annot = results.get("annotator_binary", {}).get("kappa")

        kappa_3_str = f"{kappa_3:.3f}" if kappa_3 is not None else "N/A"
        kappa_bin_str = f"{kappa_bin:.3f}" if kappa_bin is not None else "N/A"
        kappa_3_annot_str = f"{kappa_3_annot:.3f}" if kappa_3_annot is not None else "N/A"
        kappa_bin_annot_str = f"{kappa_bin_annot:.3f}" if kappa_bin_annot is not None else "N/A"

        f.write(f"| NLI vs Human | {kappa_3_str} | {kappa_bin_str} |\n")
        f.write(f"| Annotator A vs B | {kappa_3_annot_str} | {kappa_bin_annot_str} |\n\n")

        # Confusion matrix
        f.write("## Confusion Matrix (NLI rows × Human columns)\n\n")
        cm = results.get("confusion_matrix", {"labels": [], "matrix": []})
        labels = cm.get("labels", [])
        matrix = cm.get("matrix", [])

        if matrix:
            f.write("| NLI \\ Human | " + " | ".join(labels) + " |\n")
            f.write("|" + "---|" * (len(labels) + 1) + "\n")
            for i, row_label in enumerate(labels):
                row = " | ".join(str(matrix[i][j]) for j in range(len(labels)))
                f.write(f"| {row_label} | {row} |\n")
        else:
            f.write("*No human labels available for confusion matrix.*\n")
        f.write("\n")

        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        per_class = results.get("per_class", {})
        if per_class:
            f.write("| Class | Precision | Recall | F1 | Support |\n")
            f.write("|-------|-----------|--------|-----|---------|\n")
            for label, metrics in per_class.items():
                f.write(f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |\n")
        else:
            f.write("*No per-class metrics available.*\n")
        f.write("\n")

        # Raw data summary
        f.write("## Data Summary\n\n")
        f.write(f"- Total claims: {len(df)}\n")
        f.write(f"- NLI label distribution: {df['nli_label'].value_counts().to_dict()}\n")
        f.write(f"- Final human label distribution: {df['human_label'].value_counts().to_dict()}\n")
        f.write(f"- Annotator A distribution: {df['human_label_a'].value_counts().to_dict()}\n")
        f.write(f"- Annotator B distribution: {df['human_label_b'].value_counts().to_dict()}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute human-NLI agreement")
    parser.add_argument("--input", default="data/gold/human_labels_todo.csv", help="Input CSV with labels")
    parser.add_argument("--output", default="results/human_agreement.md", help="Output markdown report")
    args = parser.parse_args()

    df = load_labels(args.input)
    print(f"Loaded {len(df)} claims from {args.input}")
    print(f"NLI label dist: {df['nli_label'].value_counts().to_dict()}")
    print(f"Human A dist: {df['human_label_a'].value_counts().to_dict()}")
    print(f"Human B dist: {df['human_label_b'].value_counts().to_dict()}")
    print(f"Final human dist: {df['human_label'].value_counts().to_dict()}")

    results = compute_agreement(df)

    write_report(results, args.output, df)

    # Print PASS/FAIL
    binary_acc = results.get("nli_vs_human_binary", {}).get("accuracy")
    if binary_acc is not None:
        if binary_acc >= 0.80:
            print(f"\nPASS: Binary agreement = {binary_acc:.2%} >= 80%")
        else:
            print(f"\nFAIL: Binary agreement = {binary_acc:.2%} < 80%")
    else:
        print("\nSKIP: Binary agreement cannot be computed (human labels not filled)")

    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()