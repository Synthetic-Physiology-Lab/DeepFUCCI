"""
Find labels that disagree between independent annotations and ground truth.

This script compares masks in masks_independent/ against ground truth masks in masks/.
Labels with IoU < 0.5 are considered disagreements and are saved to a new mask image.

Usage:
    python find_disagreeing_labels.py

Output:
    For each image, saves a mask containing only the disagreeing labels to
    masks_disagreement/ folder.
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label as label_skimage
from tqdm import tqdm

# Configuration
DATA_DIR = "../../data"
DATASETS = {
    "HT1080_20x": f"{DATA_DIR}/data_set_HT1080_20x",
    "HT1080_40x": f"{DATA_DIR}/data_set_HT1080_40x",
}
IOU_THRESHOLD = 0.5


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0.0
    return intersection / union


def find_best_match_iou(label_mask, gt_mask):
    """
    Find the best matching ground truth label for a given label mask.

    Returns the maximum IoU achieved with any overlapping GT label.
    """
    # Find which GT labels overlap with this label
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


def process_image(independent_mask_path, gt_mask_path, output_path):
    """
    Compare independent annotation against ground truth and save disagreeing labels.

    Parameters
    ----------
    independent_mask_path : str
        Path to the independent annotation mask.
    gt_mask_path : str
        Path to the ground truth mask.
    output_path : str
        Path to save the disagreement mask.

    Returns
    -------
    dict
        Statistics about the comparison.
    """
    # Load masks
    independent_mask = imread(independent_mask_path)
    gt_mask = imread(gt_mask_path)

    # Ensure masks are labeled (not binary)
    if independent_mask.max() == 1:
        independent_mask = label_skimage(independent_mask)
    if gt_mask.max() == 1:
        gt_mask = label_skimage(gt_mask)

    # Get all labels in independent annotation
    independent_labels = np.unique(independent_mask)
    independent_labels = independent_labels[independent_labels > 0]

    # Track disagreeing labels
    disagreeing_labels = []
    agreeing_labels = []

    for label_id in independent_labels:
        label_mask = independent_mask == label_id
        best_iou = find_best_match_iou(label_mask, gt_mask)

        if best_iou < IOU_THRESHOLD:
            disagreeing_labels.append(label_id)
        else:
            agreeing_labels.append(label_id)

    # Create output mask with only disagreeing labels
    disagreement_mask = np.zeros_like(independent_mask)
    for i, label_id in enumerate(disagreeing_labels, start=1):
        disagreement_mask[independent_mask == label_id] = i

    # Save the disagreement mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use appropriate dtype
    if disagreement_mask.max() < 256:
        disagreement_mask = disagreement_mask.astype(np.uint8)
    else:
        disagreement_mask = disagreement_mask.astype(np.uint16)

    imsave(output_path, disagreement_mask, check_contrast=False)

    return {
        "total_labels": len(independent_labels),
        "agreeing": len(agreeing_labels),
        "disagreeing": len(disagreeing_labels),
    }


def main():
    print(f"Finding disagreeing labels (IoU threshold: {IOU_THRESHOLD})")
    print("=" * 60)

    total_stats = {
        "total_labels": 0,
        "agreeing": 0,
        "disagreeing": 0,
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")

        gt_mask_dir = f"{dataset_path}/masks"
        independent_mask_dir = f"{dataset_path}/masks_independent"
        output_dir = f"{dataset_path}/masks_disagreement"

        # Check if directories exist
        if not Path(independent_mask_dir).exists():
            print(f"  Warning: {independent_mask_dir} not found, skipping")
            continue

        if not Path(gt_mask_dir).exists():
            print(f"  Warning: {gt_mask_dir} not found, skipping")
            continue

        # Get all independent mask files
        independent_files = sorted(glob(f"{independent_mask_dir}/*.tif"))

        if not independent_files:
            print(f"  No mask files found in {independent_mask_dir}")
            continue

        print(f"  Found {len(independent_files)} images")

        dataset_stats = {
            "total_labels": 0,
            "agreeing": 0,
            "disagreeing": 0,
        }

        for independent_path in tqdm(independent_files, desc=f"  {dataset_name}"):
            filename = Path(independent_path).name
            gt_path = f"{gt_mask_dir}/{filename}"
            output_path = f"{output_dir}/{filename}"

            if not Path(gt_path).exists():
                print(f"    Warning: GT mask not found for {filename}")
                continue

            stats = process_image(independent_path, gt_path, output_path)

            for key in dataset_stats:
                dataset_stats[key] += stats[key]

        # Print dataset summary
        if dataset_stats["total_labels"] > 0:
            agreement_rate = dataset_stats["agreeing"] / dataset_stats["total_labels"] * 100
            print(f"  Results for {dataset_name}:")
            print(f"    Total labels: {dataset_stats['total_labels']}")
            print(f"    Agreeing (IoU >= {IOU_THRESHOLD}): {dataset_stats['agreeing']} ({agreement_rate:.1f}%)")
            print(f"    Disagreeing (IoU < {IOU_THRESHOLD}): {dataset_stats['disagreeing']} ({100-agreement_rate:.1f}%)")
            print(f"    Output saved to: {output_dir}")

        # Add to total
        for key in total_stats:
            total_stats[key] += dataset_stats[key]

    # Print overall summary
    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)
    if total_stats["total_labels"] > 0:
        agreement_rate = total_stats["agreeing"] / total_stats["total_labels"] * 100
        print(f"Total labels analyzed: {total_stats['total_labels']}")
        print(f"Agreeing (IoU >= {IOU_THRESHOLD}): {total_stats['agreeing']} ({agreement_rate:.1f}%)")
        print(f"Disagreeing (IoU < {IOU_THRESHOLD}): {total_stats['disagreeing']} ({100-agreement_rate:.1f}%)")
    else:
        print("No labels were analyzed. Check that mask files exist.")


if __name__ == "__main__":
    main()
