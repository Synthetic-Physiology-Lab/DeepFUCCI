"""
Analyze SNR for agreeing vs disagreeing labels between independent annotations.

This script computes the Signal-to-Noise Ratio (SNR) for nuclei that agree
(IoU >= 0.5) and disagree (IoU < 0.5) between independent annotations and
ground truth masks.

The analysis tests whether disagreements are driven by low SNR (hard to see
nuclei) or by semantic ambiguity (interpretation differences).

Usage:
    python analyze_disagreement_snr.py

Output:
    - Console summary with statistics
    - SNR histograms comparing agreeing vs disagreeing labels
    - Statistical tests (t-test, Mann-Whitney U)
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from skimage.io import imread
from skimage.measure import label as label_skimage
from tqdm import tqdm

from snr_utils import compute_background_intensity, compute_snr_for_label

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
    """Analyze SNR for agreeing vs disagreeing labels in a dataset."""
    gt_mask_dir = f"{dataset_path}/masks"
    independent_mask_dir = f"{dataset_path}/masks_independent"
    images_dir = f"{dataset_path}/images"

    if not Path(independent_mask_dir).exists():
        print(f"  Warning: {independent_mask_dir} not found, skipping")
        return None

    independent_files = sorted(glob(f"{independent_mask_dir}/*.tif"))

    if not independent_files:
        print(f"  No mask files found in {independent_mask_dir}")
        return None

    agreeing_snr = []
    disagreeing_snr = []
    per_image_results = []

    for independent_path in tqdm(independent_files, desc=f"  {dataset_name}"):
        filename = Path(independent_path).name
        gt_path = f"{gt_mask_dir}/{filename}"
        image_path = f"{images_dir}/{filename}"

        if not Path(gt_path).exists() or not Path(image_path).exists():
            continue

        # Load data
        independent_mask = imread(independent_path)
        gt_mask = imread(gt_path)
        image = imread(image_path)

        # Ensure labeled
        if independent_mask.max() == 1:
            independent_mask = label_skimage(independent_mask)
        if gt_mask.max() == 1:
            gt_mask = label_skimage(gt_mask)

        # Compute background intensity for SNR
        background_intensity = compute_background_intensity(image, independent_mask)
        if background_intensity is None:
            continue

        # Analyze each label
        independent_labels = np.unique(independent_mask)
        independent_labels = independent_labels[independent_labels > 0]

        image_agreeing = []
        image_disagreeing = []

        for label_id in independent_labels:
            label_mask = independent_mask == label_id
            best_iou = find_best_match_iou(label_mask, gt_mask)

            # Compute SNR
            snr_result = compute_snr_for_label(
                image, independent_mask, label_id, background_intensity
            )
            if snr_result is None:
                continue

            snr = snr_result["snr"]
            if isinstance(snr, list):
                snr = max(snr)  # Use max across channels

            if best_iou >= IOU_THRESHOLD:
                agreeing_snr.append(snr)
                image_agreeing.append(snr)
            else:
                disagreeing_snr.append(snr)
                image_disagreeing.append(snr)

        per_image_results.append(
            {
                "filename": filename,
                "n_agreeing": len(image_agreeing),
                "n_disagreeing": len(image_disagreeing),
                "mean_snr_agreeing": np.mean(image_agreeing) if image_agreeing else None,
                "mean_snr_disagreeing": (
                    np.mean(image_disagreeing) if image_disagreeing else None
                ),
            }
        )

    return {
        "agreeing_snr": agreeing_snr,
        "disagreeing_snr": disagreeing_snr,
        "per_image": per_image_results,
    }


def main():
    print(f"Analyzing SNR for agreeing vs disagreeing labels (IoU threshold: {IOU_THRESHOLD})")
    print("=" * 70)

    all_agreeing_snr = []
    all_disagreeing_snr = []
    dataset_results = {}

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")

        result = analyze_dataset(dataset_name, dataset_path)
        if result is None:
            continue

        dataset_results[dataset_name] = result
        all_agreeing_snr.extend(result["agreeing_snr"])
        all_disagreeing_snr.extend(result["disagreeing_snr"])

        # Print dataset summary
        agreeing = result["agreeing_snr"]
        disagreeing = result["disagreeing_snr"]
        print(f"  Agreeing: n={len(agreeing)}, mean SNR={np.mean(agreeing):.2f}, "
              f"median={np.median(agreeing):.2f}")
        print(f"  Disagreeing: n={len(disagreeing)}, mean SNR={np.mean(disagreeing):.2f}, "
              f"median={np.median(disagreeing):.2f}")

    if not all_agreeing_snr or not all_disagreeing_snr:
        print("No data to analyze.")
        return

    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    print(f"\nAgreeing labels: n={len(all_agreeing_snr)}")
    print(f"  Mean SNR: {np.mean(all_agreeing_snr):.3f}")
    print(f"  Median SNR: {np.median(all_agreeing_snr):.3f}")
    print(f"  Std SNR: {np.std(all_agreeing_snr):.3f}")
    print(f"  25th percentile: {np.percentile(all_agreeing_snr, 25):.3f}")
    print(f"  75th percentile: {np.percentile(all_agreeing_snr, 75):.3f}")

    print(f"\nDisagreeing labels: n={len(all_disagreeing_snr)}")
    print(f"  Mean SNR: {np.mean(all_disagreeing_snr):.3f}")
    print(f"  Median SNR: {np.median(all_disagreeing_snr):.3f}")
    print(f"  Std SNR: {np.std(all_disagreeing_snr):.3f}")
    print(f"  25th percentile: {np.percentile(all_disagreeing_snr, 25):.3f}")
    print(f"  75th percentile: {np.percentile(all_disagreeing_snr, 75):.3f}")

    # Statistical tests
    print("\n" + "-" * 70)
    print("Statistical Tests")
    print("-" * 70)

    t_stat, p_ttest = stats.ttest_ind(all_agreeing_snr, all_disagreeing_snr)
    print(f"T-test (agreeing vs disagreeing): t={t_stat:.3f}, p={p_ttest:.2e}")

    u_stat, p_mw = stats.mannwhitneyu(
        all_agreeing_snr, all_disagreeing_snr, alternative="two-sided"
    )
    print(f"Mann-Whitney U test: U={u_stat:.0f}, p={p_mw:.2e}")

    # Interpretation
    alpha = 0.05
    if p_ttest > alpha and p_mw > alpha:
        print(f"\nConclusion: No significant difference in SNR (p > {alpha})")
        print("Disagreements are NOT driven by low SNR (hard to see nuclei).")
        print("Disagreements likely reflect semantic ambiguity in annotation.")
    else:
        print(f"\nConclusion: Significant difference in SNR detected (p < {alpha})")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(
        min(min(all_agreeing_snr), min(all_disagreeing_snr)),
        max(max(all_agreeing_snr), max(all_disagreeing_snr)),
        30,
    )
    ax.hist(all_agreeing_snr, bins=bins, alpha=0.6, label=f"Agreeing (n={len(all_agreeing_snr)})", density=True)
    ax.hist(all_disagreeing_snr, bins=bins, alpha=0.6, label=f"Disagreeing (n={len(all_disagreeing_snr)})", density=True)
    ax.axvline(np.mean(all_agreeing_snr), color="C0", linestyle="--", label=f"Mean agreeing: {np.mean(all_agreeing_snr):.2f}")
    ax.axvline(np.mean(all_disagreeing_snr), color="C1", linestyle="--", label=f"Mean disagreeing: {np.mean(all_disagreeing_snr):.2f}")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Density")
    ax.set_title("SNR Distribution: Agreeing vs Disagreeing Labels")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    data = [all_agreeing_snr, all_disagreeing_snr]
    bp = ax.boxplot(data, labels=["Agreeing", "Disagreeing"], patch_artist=True)
    bp["boxes"][0].set_facecolor("C0")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("C1")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_ylabel("SNR")
    ax.set_title(f"SNR Comparison (p={p_mw:.3f})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("disagreement_snr_analysis.pdf")
    plt.savefig("disagreement_snr_analysis.png", dpi=150)
    print("Saved: disagreement_snr_analysis.pdf/png")
    plt.close()

    # Save results to JSON
    output_data = {
        "iou_threshold": IOU_THRESHOLD,
        "overall": {
            "agreeing": {
                "n": len(all_agreeing_snr),
                "mean_snr": float(np.mean(all_agreeing_snr)),
                "median_snr": float(np.median(all_agreeing_snr)),
                "std_snr": float(np.std(all_agreeing_snr)),
                "percentile_25": float(np.percentile(all_agreeing_snr, 25)),
                "percentile_75": float(np.percentile(all_agreeing_snr, 75)),
            },
            "disagreeing": {
                "n": len(all_disagreeing_snr),
                "mean_snr": float(np.mean(all_disagreeing_snr)),
                "median_snr": float(np.median(all_disagreeing_snr)),
                "std_snr": float(np.std(all_disagreeing_snr)),
                "percentile_25": float(np.percentile(all_disagreeing_snr, 25)),
                "percentile_75": float(np.percentile(all_disagreeing_snr, 75)),
            },
            "statistics": {
                "ttest_t": float(t_stat),
                "ttest_p": float(p_ttest),
                "mannwhitney_u": float(u_stat),
                "mannwhitney_p": float(p_mw),
            },
        },
        "per_dataset": {},
    }

    for dataset_name, result in dataset_results.items():
        agreeing = result["agreeing_snr"]
        disagreeing = result["disagreeing_snr"]
        output_data["per_dataset"][dataset_name] = {
            "agreeing": {
                "n": len(agreeing),
                "mean_snr": float(np.mean(agreeing)),
                "median_snr": float(np.median(agreeing)),
            },
            "disagreeing": {
                "n": len(disagreeing),
                "mean_snr": float(np.mean(disagreeing)) if disagreeing else None,
                "median_snr": float(np.median(disagreeing)) if disagreeing else None,
            },
        }

    with open("disagreement_snr_results.json", "w") as fp:
        json.dump(output_data, fp, indent=2)
    print("Saved: disagreement_snr_results.json")


if __name__ == "__main__":
    main()
