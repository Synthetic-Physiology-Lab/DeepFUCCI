"""
Analyze segmentation accuracy as a function of Signal-to-Noise Ratio (SNR)
for DAPI-equivalent StarDist methods (with and without postprocessing).

This script:
1. Computes SNR for each ground truth nucleus
2. Runs both DAPI-equivalent prediction methods:
   - Raw: normalize + max projection
   - Postprocessed: top-hat + blur + normalize + max projection
3. Matches predictions to ground truth
4. Analyzes detection rate (recall) binned by SNR

SNR = (I_in - I_out) / std_in

Usage:
    python snr_accuracy_dapieq_analysis.py
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyclesperanto_prototype as cle
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.measure import label as label_skimage
from stardist import fill_label_holes, gputools_available
from stardist.models import StarDist2D
from tqdm import tqdm

from snr_utils import compute_background_intensity, compute_snr_for_label

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "../../data"
TRAINING_DATA_DIR = f"{DATA_DIR}/training_data_tiled_strict_classified"
AXIS_NORM = (0, 1)
IOU_THRESHOLD = 0.5
SNR_BINS = 10

# DAPI-equivalent preprocessing parameters
NUCLEUS_RADIUS_MICRONS = 10.0
PIXEL_SIZE_MICRONS = 0.3
NUCLEUS_RADIUS_PIXEL = NUCLEUS_RADIUS_MICRONS / PIXEL_SIZE_MICRONS

# External test datasets
EXTERNAL_DATASETS = {
    "ConfluentFUCCI": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/test_confluentfucci/images",
        "masks_dir": f"{DATA_DIR}/test_confluentfucci/masks",
        "relabel": True,
        "channel_order": "YXC",
    },
    "HT1080_20x": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/data_set_HT1080_20x/images",
        "masks_dir": f"{DATA_DIR}/data_set_HT1080_20x/masks",
        "relabel": True,
        "channel_order": "YXC",
    },
    "HT1080_40x": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/data_set_HT1080_40x/images",
        "masks_dir": f"{DATA_DIR}/data_set_HT1080_40x/masks",
        "relabel": True,
        "channel_order": "YXC",
    },
    "CellMAPtracer": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/test_cellmaptracer/images",
        "masks_dir": f"{DATA_DIR}/test_cellmaptracer/masks",
        "relabel": True,
        "channel_order": "CYX",
        "pixel_size": 0.65,
    },
    "CottonEtAl": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/test_cottonetal/images",
        "masks_dir": f"{DATA_DIR}/test_cottonetal/masks",
        "relabel": True,
        "channel_order": "CYX",
        "pixel_size": 0.67,
    },
    "HaCaT_Han": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/HaCaT_Han_et_al/images",
        "masks_dir": f"{DATA_DIR}/HaCaT_Han_et_al/masks",
        "relabel": True,
        "channel_order": "YXC",
    },
}

# Methods to analyze
METHODS = {
    "raw": "DAPI-eq (raw)",
    "postprocessed": "DAPI-eq (postprocessed)",
}


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_dataset_with_split(data_dir, split_key):
    """Load images and masks for a specific split."""
    split_file = Path(data_dir) / "dataset_split.json"
    with open(split_file) as fp:
        dataset_split = json.load(fp)

    filenames = dataset_split[split_key]
    images = [
        imread(f"{data_dir}/images/{fname}")
        for fname in tqdm(filenames, desc=f"Loading {split_key} images")
    ]
    masks = [
        fill_label_holes(imread(f"{data_dir}/masks/{fname}"))
        for fname in tqdm(filenames, desc=f"Loading {split_key} masks")
    ]

    return images, masks, filenames


def load_directory_dataset(images_dir, masks_dir, relabel=False, channel_order="YXC"):
    """Load dataset from directory."""
    image_files = sorted(glob(f"{images_dir}/*.tif"))
    mask_files = sorted(glob(f"{masks_dir}/*.tif"))

    if not image_files:
        return None, None, None

    assert len(image_files) == len(mask_files)
    assert all(Path(x).name == Path(y).name for x, y in zip(image_files, mask_files))

    images = []
    for f in tqdm(image_files, desc="Loading images"):
        img = imread(f)
        if channel_order == "CYX" and img.ndim == 3:
            img = np.moveaxis(img, 0, -1)
        images.append(img)

    masks = []
    for f in tqdm(mask_files, desc="Loading masks"):
        mask = imread(f)
        if relabel:
            mask = label_skimage(mask)
        masks.append(fill_label_holes(mask))

    filenames = [Path(f).name for f in image_files]
    return images, masks, filenames


def load_external_dataset(config):
    """Load external dataset based on config."""
    dataset_type = config.get("type", "directory")
    relabel = config.get("relabel", False)
    channel_order = config.get("channel_order", "YXC")

    if dataset_type == "directory":
        return load_directory_dataset(
            config["images_dir"],
            config["masks_dir"],
            relabel=relabel,
            channel_order=channel_order,
        )
    return None, None, None


# =============================================================================
# DAPI-Equivalent Prediction Functions
# =============================================================================


def dapieq_predict_raw(model, x):
    """
    DAPI-equivalent prediction without postprocessing (raw).
    Just normalize + max projection.
    """
    ch1 = x[..., 0]
    ch2 = x[..., 1]

    normal_ch1 = normalize(ch1)
    normal_ch2 = normalize(ch2)

    max_projected = np.maximum(normal_ch1, normal_ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


def dapieq_predict_postprocessed(model, x, nucleus_radius_pixel):
    """
    DAPI-equivalent prediction with postprocessing.
    Top-hat + blur + normalize + max projection.
    """
    ch1 = x[..., 0]
    ch2 = x[..., 1]

    # Top-hat filtering and blur
    ch1_top = cle.top_hat_sphere(
        ch1, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel
    )
    ch1_blur = cle.gaussian_blur(ch1_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch1 = normalize(ch1_blur.get())

    ch2_top = cle.top_hat_sphere(
        ch2, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel
    )
    ch2_blur = cle.gaussian_blur(ch2_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch2 = normalize(ch2_blur.get())

    max_projected = np.maximum(normal_ch1, normal_ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


# =============================================================================
# SNR and Accuracy Analysis Functions
# =============================================================================


def compute_snr_per_nucleus(images, masks):
    """Compute SNR for each nucleus."""
    snr_data = {}

    for img_idx, (img, mask) in enumerate(
        tqdm(zip(images, masks), total=len(images), desc="Computing SNR")
    ):
        background_intensity = compute_background_intensity(img, mask)
        if background_intensity is None:
            continue

        labels = np.unique(mask)
        labels = labels[labels > 0]

        for label_id in labels:
            result = compute_snr_for_label(img, mask, label_id, background_intensity)
            if result is not None:
                snr = result["snr"]
                if isinstance(snr, list):
                    snr = max(snr)
                snr_data[(img_idx, int(label_id))] = snr

    return snr_data


def analyze_accuracy_vs_snr(images, masks, predictions, snr_data, iou_threshold=0.5):
    """Analyze detection accuracy as a function of SNR."""
    nucleus_results = []
    total_tp = 0
    total_fn = 0
    total_fp = 0

    for img_idx, (mask_gt, mask_pred) in enumerate(zip(masks, predictions)):
        gt_labels = np.unique(mask_gt)
        gt_labels = gt_labels[gt_labels > 0]
        pred_labels_all = np.unique(mask_pred)
        pred_labels_all = pred_labels_all[pred_labels_all > 0]

        matched_predictions = set()

        for gt_label in gt_labels:
            key = (img_idx, int(gt_label))
            snr = snr_data.get(key, None)

            gt_mask = mask_gt == gt_label
            detected = False
            best_pred = None

            pred_labels = np.unique(mask_pred[gt_mask])
            pred_labels = pred_labels[pred_labels > 0]

            for pred_label in pred_labels:
                pred_mask = mask_pred == pred_label
                intersection = np.sum(gt_mask & pred_mask)
                union = np.sum(gt_mask | pred_mask)
                iou = intersection / union if union > 0 else 0

                if iou >= iou_threshold:
                    detected = True
                    best_pred = pred_label
                    break

            if detected:
                total_tp += 1
                if best_pred is not None:
                    matched_predictions.add(best_pred)
            else:
                total_fn += 1

            if snr is not None:
                nucleus_results.append((snr, detected))

        for pred_label in pred_labels_all:
            if pred_label not in matched_predictions:
                total_fp += 1

    overall_metrics = {
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        "accuracy": total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) > 0 else 0,
    }

    return nucleus_results, overall_metrics


def bin_and_compute_recall(nucleus_results, n_bins=10):
    """Bin nuclei by SNR and compute recall per bin."""
    if not nucleus_results:
        return [], [], [], []

    snr_values = np.array([r[0] for r in nucleus_results])
    detected = np.array([r[1] for r in nucleus_results])

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(snr_values, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return [], [], [], []

    bin_centers = []
    recalls = []
    counts = []

    for i in range(len(bin_edges) - 1):
        mask = (snr_values >= bin_edges[i]) & (snr_values < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            mask = (snr_values >= bin_edges[i]) & (snr_values <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            recalls.append(detected[mask].mean())
            counts.append(mask.sum())

    return bin_centers, recalls, counts, bin_edges


def run_predictions(model, images_norm, method, nucleus_radius_pixel):
    """Run predictions for a given method."""
    predictions = []
    desc = f"Predicting ({method})"
    for x in tqdm(images_norm, desc=desc):
        if method == "raw":
            pred_mask = dapieq_predict_raw(model, x)
        else:
            pred_mask = dapieq_predict_postprocessed(model, x, nucleus_radius_pixel)
        predictions.append(pred_mask)
    return predictions


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_per_dataset(all_results, method_name, output_prefix):
    """Plot detection rate vs SNR for each dataset (one method)."""
    n_datasets = len(all_results)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )

    dataset_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (dataset_name, data) in enumerate(all_results.items()):
        ax = axes[idx // n_cols, idx % n_cols]

        if data["bin_centers"]:
            ax.plot(
                data["bin_centers"],
                data["recalls"],
                marker="o",
                color=dataset_colors[idx],
                linewidth=2,
                markersize=6,
            )
            # Add count labels (just the number)
            for bc, rc, cnt in zip(data["bin_centers"], data["recalls"], data["counts"]):
                ax.annotate(
                    f"{cnt}",
                    (bc, rc),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7,
                    alpha=0.7,
                )

        ax.set_xlabel("SNR")
        ax.set_ylabel("Detection Rate (Recall)")
        ax.set_title(
            f"{dataset_name}\n"
            f"R={data['overall_recall']:.3f}, P={data['overall_precision']:.3f}, "
            f"Acc={data['overall_accuracy']:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    for idx in range(n_datasets, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.suptitle(f"{method_name} - SNR vs Detection Rate", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_by_dataset.pdf")
    plt.savefig(f"{output_prefix}_by_dataset.png", dpi=150)
    print(f"Saved: {output_prefix}_by_dataset.pdf/png")
    plt.close()


def plot_comparison(all_results, method_name, output_prefix):
    """Plot comparison across datasets (one method)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (dataset_name, data) in enumerate(all_results.items()):
        if data["bin_centers"]:
            ax.plot(
                data["bin_centers"],
                data["recalls"],
                marker="o",
                color=dataset_colors[idx],
                label=f"{dataset_name} (R={data['overall_recall']:.2f})",
                linewidth=2,
                markersize=6,
            )

    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Detection Rate (Recall)")
    ax.set_title(f"{method_name} - Detection Rate vs SNR (IoU={IOU_THRESHOLD})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.pdf")
    plt.savefig(f"{output_prefix}_comparison.png", dpi=150)
    print(f"Saved: {output_prefix}_comparison.pdf/png")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    use_gpu = gputools_available()
    print(f"Using GPU: {use_gpu}")

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.1, total_memory=50000)

    # Load model
    print("\nLoading pretrained StarDist model (2D_versatile_fluo)...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    print("Model loaded.")

    # Load datasets
    all_datasets = {}

    print("\nLoading validation data...")
    if Path(f"{TRAINING_DATA_DIR}/dataset_split.json").exists():
        X_val, Y_val, _ = load_dataset_with_split(TRAINING_DATA_DIR, "validation")
        all_datasets["Validation"] = {
            "images": X_val,
            "masks": Y_val,
            "pixel_size": PIXEL_SIZE_MICRONS,
        }

    print("\nLoading external datasets...")
    for name, config in EXTERNAL_DATASETS.items():
        print(f"  Trying to load {name}...")
        try:
            X_ext, Y_ext, _ = load_external_dataset(config)
            if X_ext is not None and len(X_ext) > 0:
                all_datasets[name] = {
                    "images": X_ext,
                    "masks": Y_ext,
                    "pixel_size": config.get("pixel_size", PIXEL_SIZE_MICRONS),
                }
                print(f"    Loaded {len(X_ext)} images")
            else:
                print("    Not found or empty")
        except Exception as e:
            print(f"    Error: {e}")

    if not all_datasets:
        print("No datasets found.")
        return

    print(f"\nLoaded {len(all_datasets)} datasets: {list(all_datasets.keys())}")

    # Store results for both methods
    results_by_method = {method: {} for method in METHODS.keys()}

    # Process each dataset
    for dataset_name, dataset_info in all_datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 60}")

        images = dataset_info["images"]
        masks = dataset_info["masks"]
        pixel_size = dataset_info["pixel_size"]
        nucleus_radius_pixel = NUCLEUS_RADIUS_MICRONS / pixel_size

        print(f"  Pixel size: {pixel_size} um, nucleus radius: {nucleus_radius_pixel:.1f} px")

        # Normalize images
        print("Normalizing images...")
        images_norm = [
            normalize(x[..., 0:2], 1, 99.8, axis=AXIS_NORM)
            for x in tqdm(images)
        ]

        # Compute SNR (same for both methods)
        print("Computing SNR...")
        snr_data = compute_snr_per_nucleus(images, masks)
        print(f"  Computed SNR for {len(snr_data)} nuclei")

        # Run predictions for each method
        for method_key, method_label in METHODS.items():
            print(f"\nRunning {method_label}...")
            predictions = run_predictions(model, images_norm, method_key, nucleus_radius_pixel)

            nucleus_results, overall_metrics = analyze_accuracy_vs_snr(
                images, masks, predictions, snr_data, iou_threshold=IOU_THRESHOLD
            )

            bin_centers, recalls, counts, bin_edges = bin_and_compute_recall(
                nucleus_results, n_bins=SNR_BINS
            )

            print(f"  TP={overall_metrics['tp']}, FN={overall_metrics['fn']}, FP={overall_metrics['fp']}")
            print(f"  Recall: {overall_metrics['recall']:.3f}, Precision: {overall_metrics['precision']:.3f}")
            print(f"  Accuracy: {overall_metrics['accuracy']:.3f}")

            snr_vals = [r[0] for r in nucleus_results]
            results_by_method[method_key][dataset_name] = {
                "nucleus_results": nucleus_results,
                "bin_centers": bin_centers,
                "recalls": recalls,
                "counts": counts,
                "overall_recall": overall_metrics["recall"],
                "overall_precision": overall_metrics["precision"],
                "overall_accuracy": overall_metrics["accuracy"],
                "tp": overall_metrics["tp"],
                "fn": overall_metrics["fn"],
                "fp": overall_metrics["fp"],
                "snr_mean": float(np.mean(snr_vals)) if snr_vals else None,
                "snr_std": float(np.std(snr_vals)) if snr_vals else None,
            }

    # Generate plots for each method
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    for method_key, method_label in METHODS.items():
        output_prefix = f"snr_accuracy_dapieq_{method_key}"
        plot_per_dataset(results_by_method[method_key], method_label, output_prefix)
        plot_comparison(results_by_method[method_key], method_label, output_prefix)

    # Save results to JSON
    output_data = {}
    for method_key, method_label in METHODS.items():
        output_data[method_key] = {"method_label": method_label, "datasets": {}}
        for dataset_name, data in results_by_method[method_key].items():
            output_data[method_key]["datasets"][dataset_name] = {
                "n_nuclei": len(data["nucleus_results"]),
                "tp": int(data["tp"]),
                "fn": int(data["fn"]),
                "fp": int(data["fp"]),
                "overall_recall": float(data["overall_recall"]),
                "overall_precision": float(data["overall_precision"]),
                "overall_accuracy": float(data["overall_accuracy"]),
                "bin_centers": [float(x) for x in data["bin_centers"]],
                "recalls_per_bin": [float(x) for x in data["recalls"]],
                "counts_per_bin": [int(x) for x in data["counts"]],
                "snr_mean": data["snr_mean"],
                "snr_std": data["snr_std"],
            }

    with open("snr_accuracy_dapieq_results.json", "w") as fp:
        json.dump(output_data, fp, indent=2)
    print("Saved: snr_accuracy_dapieq_results.json")

    # Print summary table
    print("\n" + "=" * 90)
    print("Summary: DAPI-Equivalent Methods Detection Metrics")
    print("=" * 90)
    for method_key, method_label in METHODS.items():
        print(f"\n{method_label}:")
        print(f"{'Dataset':<20} {'Recall':>10} {'Precision':>10} {'Accuracy':>10} {'SNR mean':>10}")
        print("-" * 60)
        for dataset_name, data in results_by_method[method_key].items():
            snr_str = f"{data['snr_mean']:.2f}" if data['snr_mean'] else "N/A"
            print(
                f"{dataset_name:<20} "
                f"{data['overall_recall']:>10.3f} "
                f"{data['overall_precision']:>10.3f} "
                f"{data['overall_accuracy']:>10.3f} "
                f"{snr_str:>10}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
