"""
Analyze IoU distribution of disagreeing labels between independent annotations.

This script categorizes disagreements by their IoU values to understand
the nature of inter-annotator disagreements:
- IoU = 0: Extra labels or completely missed cells
- IoU 0-0.3: Likely merged/split cells
- IoU 0.3-0.5: Boundary disagreements

Usage:
    python analyze_disagreement_iou.py

Output:
    - Console summary with statistics
    - IoU distribution histogram
    - disagreement_iou_results.json
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.measure import label as label_skimage
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "../../data"
DATASETS = {
    "HT1080_20x": f"{DATA_DIR}/data_set_HT1080_20x",
    "HT1080_40x": f"{DATA_DIR}/data_set_HT1080_40x",
}
IOU_THRESHOLD = 0.5


# =============================================================================
# Helper Functions
# =============================================================================


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0.0


def find_best_match_iou(label_mask, gt_mask):
    """
    Find the best matching ground truth label for a given label mask.
    Returns the maximum IoU achieved with any overlapping GT label.
    """
    overlapping_gt_labels = np.unique(gt_mask[label_mask])
    overlapping_gt_labels = overlapping_gt_labels[overlapping_gt_labels > 0]

    if len(overlapping_gt_labels) == 0:
        return 0.0

    best_iou = 0.0
    for gt_label in overlapping_gt_labels:
        gt_label_mask = gt_mask == gt_label
        iou = compute_iou(label_mask, gt_label_mask)
        best_iou = max(best_iou, iou)

    return best_iou


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_dataset(dataset_name, dataset_path):
    """Analyze IoU distribution for a dataset."""
    gt_mask_dir = f"{dataset_path}/masks"
    independent_mask_dir = f"{dataset_path}/masks_independent"

    if not Path(independent_mask_dir).exists():
        print(f"  Warning: {independent_mask_dir} not found, skipping")
        return None

    independent_files = sorted(glob(f"{independent_mask_dir}/*.tif"))

    if not independent_files:
        print(f"  No mask files found in {independent_mask_dir}")
        return None

    iou_agreeing = []
    iou_disagreeing = []

    for independent_path in tqdm(independent_files, desc=f"  {dataset_name}"):
        filename = Path(independent_path).name
        gt_path = f"{gt_mask_dir}/{filename}"

        if not Path(gt_path).exists():
            continue

        independent_mask = imread(independent_path)
        gt_mask = imread(gt_path)

        if independent_mask.max() == 1:
            independent_mask = label_skimage(independent_mask)
        if gt_mask.max() == 1:
            gt_mask = label_skimage(gt_mask)

        independent_labels = np.unique(independent_mask)
        independent_labels = independent_labels[independent_labels > 0]

        for label_id in independent_labels:
            label_mask = independent_mask == label_id
            best_iou = find_best_match_iou(label_mask, gt_mask)

            if best_iou >= IOU_THRESHOLD:
                iou_agreeing.append(best_iou)
            else:
                iou_disagreeing.append(best_iou)

    return {
        "agreeing": iou_agreeing,
        "disagreeing": iou_disagreeing,
    }


def categorize_disagreements(iou_values):
    """Categorize disagreements by IoU range."""
    iou = np.array(iou_values)
    total = len(iou)

    if total == 0:
        return {}

    categories = {
        "no_overlap": {
            "range": "IoU = 0",
            "description": "Extra labels or completely missed cells",
            "count": int(np.sum(iou == 0)),
        },
        "very_low": {
            "range": "0 < IoU < 0.1",
            "description": "Minimal overlap",
            "count": int(np.sum((iou > 0) & (iou < 0.1))),
        },
        "low": {
            "range": "0.1 <= IoU < 0.2",
            "description": "Low overlap",
            "count": int(np.sum((iou >= 0.1) & (iou < 0.2))),
        },
        "partial": {
            "range": "0.2 <= IoU < 0.3",
            "description": "Partial overlap - likely merged/split",
            "count": int(np.sum((iou >= 0.2) & (iou < 0.3))),
        },
        "moderate": {
            "range": "0.3 <= IoU < 0.4",
            "description": "Moderate overlap - boundary differences",
            "count": int(np.sum((iou >= 0.3) & (iou < 0.4))),
        },
        "near_match": {
            "range": "0.4 <= IoU < 0.5",
            "description": "Near match - slight boundary disagreement",
            "count": int(np.sum((iou >= 0.4) & (iou < 0.5))),
        },
    }

    for cat in categories.values():
        cat["percentage"] = 100 * cat["count"] / total

    # Summary groups
    categories["summary"] = {
        "no_overlap_total": {
            "range": "IoU = 0",
            "count": int(np.sum(iou == 0)),
            "percentage": 100 * np.sum(iou == 0) / total,
        },
        "partial_overlap_total": {
            "range": "0 < IoU < 0.3",
            "count": int(np.sum((iou > 0) & (iou < 0.3))),
            "percentage": 100 * np.sum((iou > 0) & (iou < 0.3)) / total,
        },
        "near_match_total": {
            "range": "0.3 <= IoU < 0.5",
            "count": int(np.sum(iou >= 0.3)),
            "percentage": 100 * np.sum(iou >= 0.3) / total,
        },
    }

    return categories


def main():
    print(f"Analyzing IoU distribution of disagreeing labels (threshold: {IOU_THRESHOLD})")
    print("=" * 70)

    all_iou_agreeing = []
    all_iou_disagreeing = []
    dataset_results = {}

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")

        result = analyze_dataset(dataset_name, dataset_path)
        if result is None:
            continue

        dataset_results[dataset_name] = result
        all_iou_agreeing.extend(result["agreeing"])
        all_iou_disagreeing.extend(result["disagreeing"])

    if not all_iou_disagreeing:
        print("No disagreeing labels found.")
        return

    disagreeing = np.array(all_iou_disagreeing)
    agreeing = np.array(all_iou_agreeing)
    total = len(disagreeing)

    # Categorize
    categories = categorize_disagreements(all_iou_disagreeing)

    # Print results
    print("\n" + "=" * 70)
    print("IoU Distribution of Disagreeing Labels")
    print("=" * 70)

    print(f"\nTotal disagreeing labels: {total}")
    print(f"\nIoU Range Breakdown:")
    for key in ["no_overlap", "very_low", "low", "partial", "moderate", "near_match"]:
        cat = categories[key]
        print(f"  {cat['range']:20s}: {cat['count']:4d} ({cat['percentage']:5.1f}%)")

    print(f"\nSummary Statistics:")
    print(f"  Mean IoU:   {np.mean(disagreeing):.3f}")
    print(f"  Median IoU: {np.median(disagreeing):.3f}")
    print(f"  Std IoU:    {np.std(disagreeing):.3f}")
    print(f"  Min IoU:    {np.min(disagreeing):.3f}")
    print(f"  Max IoU:    {np.max(disagreeing):.3f}")

    print(f"\nDisagreement Types (Summary):")
    summary = categories["summary"]
    print(f"  No overlap (IoU=0):           {summary['no_overlap_total']['count']:4d} "
          f"({summary['no_overlap_total']['percentage']:5.1f}%) - Extra or missed cells")
    print(f"  Partial overlap (0<IoU<0.3):  {summary['partial_overlap_total']['count']:4d} "
          f"({summary['partial_overlap_total']['percentage']:5.1f}%) - Merged/split cells")
    print(f"  Near match (0.3<=IoU<0.5):    {summary['near_match_total']['count']:4d} "
          f"({summary['near_match_total']['percentage']:5.1f}%) - Boundary disagreements")

    # Per dataset
    print("\n" + "-" * 70)
    print("Per Dataset Breakdown")
    print("-" * 70)

    for dataset_name, data in dataset_results.items():
        d = np.array(data["disagreeing"])
        if len(d) == 0:
            continue
        print(f"\n{dataset_name}: n={len(d)}")
        print(f"  IoU=0: {np.sum(d == 0)} ({100 * np.sum(d == 0) / len(d):.1f}%)")
        print(f"  Mean IoU: {np.mean(d):.3f}, Median: {np.median(d):.3f}")

    # Agreeing labels for reference
    print("\n" + "-" * 70)
    print("Agreeing Labels (for reference)")
    print("-" * 70)
    print(f"  n={len(agreeing)}")
    print(f"  Mean IoU: {np.mean(agreeing):.3f}")
    print(f"  Median IoU: {np.median(agreeing):.3f}")
    print(f"  Min IoU: {np.min(agreeing):.3f}")
    print(f"  Max IoU: {np.max(agreeing):.3f}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Histogram of disagreeing IoU values
    ax = axes[0]
    bins = np.linspace(0, 0.5, 21)
    ax.hist(disagreeing, bins=bins, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(disagreeing), color="red", linestyle="--",
               label=f"Mean: {np.mean(disagreeing):.3f}")
    ax.axvline(np.median(disagreeing), color="orange", linestyle="--",
               label=f"Median: {np.median(disagreeing):.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution of Disagreeing Labels")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Stacked bar chart of categories
    ax = axes[1]
    cat_names = ["No overlap\n(IoU=0)", "Partial\n(0<IoU<0.3)", "Near match\n(0.3≤IoU<0.5)"]
    cat_counts = [
        summary["no_overlap_total"]["count"],
        summary["partial_overlap_total"]["count"],
        summary["near_match_total"]["count"],
    ]
    cat_pcts = [
        summary["no_overlap_total"]["percentage"],
        summary["partial_overlap_total"]["percentage"],
        summary["near_match_total"]["percentage"],
    ]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(cat_names, cat_counts, color=colors, edgecolor="black")
    for bar, pct in zip(bars, cat_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Disagreement Types")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Comparison of agreeing vs disagreeing IoU
    ax = axes[2]
    ax.hist(agreeing, bins=np.linspace(0, 1, 31), alpha=0.6,
            label=f"Agreeing (n={len(agreeing)})", density=True)
    ax.hist(disagreeing, bins=np.linspace(0, 1, 31), alpha=0.6,
            label=f"Disagreeing (n={len(disagreeing)})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", label="Threshold (0.5)")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Density")
    ax.set_title("IoU Distribution: Agreeing vs Disagreeing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("disagreement_iou_analysis.pdf")
    plt.savefig("disagreement_iou_analysis.png", dpi=150)
    print("Saved: disagreement_iou_analysis.pdf/png")
    plt.close()

    # Save results to JSON
    output_data = {
        "iou_threshold": IOU_THRESHOLD,
        "overall": {
            "total_disagreeing": total,
            "total_agreeing": len(agreeing),
            "disagreeing_stats": {
                "mean_iou": float(np.mean(disagreeing)),
                "median_iou": float(np.median(disagreeing)),
                "std_iou": float(np.std(disagreeing)),
                "min_iou": float(np.min(disagreeing)),
                "max_iou": float(np.max(disagreeing)),
            },
            "agreeing_stats": {
                "mean_iou": float(np.mean(agreeing)),
                "median_iou": float(np.median(agreeing)),
                "std_iou": float(np.std(agreeing)),
                "min_iou": float(np.min(agreeing)),
                "max_iou": float(np.max(agreeing)),
            },
            "categories": {
                "no_overlap": {
                    "count": summary["no_overlap_total"]["count"],
                    "percentage": summary["no_overlap_total"]["percentage"],
                },
                "partial_overlap": {
                    "count": summary["partial_overlap_total"]["count"],
                    "percentage": summary["partial_overlap_total"]["percentage"],
                },
                "near_match": {
                    "count": summary["near_match_total"]["count"],
                    "percentage": summary["near_match_total"]["percentage"],
                },
            },
        },
        "per_dataset": {},
    }

    for dataset_name, data in dataset_results.items():
        d = np.array(data["disagreeing"])
        a = np.array(data["agreeing"])
        output_data["per_dataset"][dataset_name] = {
            "n_disagreeing": len(d),
            "n_agreeing": len(a),
            "disagreeing_mean_iou": float(np.mean(d)) if len(d) > 0 else None,
            "disagreeing_median_iou": float(np.median(d)) if len(d) > 0 else None,
            "no_overlap_count": int(np.sum(d == 0)) if len(d) > 0 else 0,
            "no_overlap_percentage": float(100 * np.sum(d == 0) / len(d)) if len(d) > 0 else 0,
        }

    with open("disagreement_iou_results.json", "w") as fp:
        json.dump(output_data, fp, indent=2)
    print("Saved: disagreement_iou_results.json")


if __name__ == "__main__":
    main()
