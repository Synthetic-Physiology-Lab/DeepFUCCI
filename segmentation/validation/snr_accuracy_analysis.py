"""
Analyze segmentation accuracy as a function of Signal-to-Noise Ratio (SNR).

This script:
1. Computes SNR for each ground truth nucleus
2. Runs segmentation predictions
3. Matches predictions to ground truth
4. Analyzes detection rate (recall) binned by SNR

Supports training/validation/test data and external test datasets.

SNR = (I_in - I_out) / std_in
Reference: https://imagej.net/plugins/trackmate/analyzers/#contrast--signalnoise-ratio

IMPORTANT: For FUCCI data with 3 channels, only the nuclear channels are used for SNR:
- Channel 0: Cyan (G1 phase marker) - NUCLEAR
- Channel 1: Magenta (S/G2/M phase marker) - NUCLEAR
- Channel 2: Tubulin - CYTOPLASMIC (EXCLUDED from SNR analysis)

The tubulin channel is cytoplasmic and has no nuclear signal, so including it
would distort the SNR analysis.

Usage:
    python snr_accuracy_analysis.py

Requires:
    - training_data/ folder with images/, masks/, and dataset_split.json
    - Trained StarDist models in the expected locations
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
IOU_THRESHOLD = 0.5  # IoU threshold for matching
SNR_BINS = 10  # Number of bins for SNR analysis

# External test datasets configuration
# Supports two types:
#   - "directory": loads all .tif files from images_dir and masks_dir
#   - "single_files": loads specific image and mask files from lists
EXTERNAL_DATASETS = {
    "ConfluentFUCCI": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/test_confluent_fucci_data/images",
        "masks_dir": f"{DATA_DIR}/test_confluent_fucci_data/masks",
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
        "type": "single_files",
        "image_files": [f"{DATA_DIR}/test_cellmaptracer/image_cyan_magenta_last_frame.tif"],
        "mask_files": [f"{DATA_DIR}/test_cellmaptracer/gt_last_frame.tif"],
        "relabel": True,
        "channel_order": "CYX",
        "scale": 0.65 / 0.34,  # pixel size correction
    },
    "CottonEtAl": {
        "type": "single_files",
        "image_files": [f"{DATA_DIR}/test_cottonetal/frame_69.tif"],
        "mask_files": [f"{DATA_DIR}/test_cottonetal/gt_frame_69.tif"],
        "relabel": True,
        "channel_order": "CYX",
        "scale": 2.0,  # pixel size correction
    },
    "HaCaT_Han": {
        "type": "ome_tiff",
        "data_dir": f"{DATA_DIR}/HaCaT_Han_et_al",
        "image_file": "merged.ome.tif",
        "mask_file": "labels_manual_annotation.tif",
        "metadata_file": "metadata.yml",
        "relabel": True,
    },
}

# Model configurations
MODEL_CONFIGS = {
    "1 CH (tubulin)": {
        "name": "stardist_1_channel_latest",
        "basedir": Path.home() / "models",
        "input_fn": lambda x: x[..., 2] if x.ndim == 3 else x,
    },
    "2 CH (cyan+magenta)": {
        "name": "stardist_2_channel_latest",
        "basedir": Path.home() / "models",
        "input_fn": lambda x: x[..., 0:2] if x.ndim == 3 else x,
    },
    "3 CH (all)": {
        "name": "stardist_3_channel_latest",
        "basedir": Path.home() / "models",
        "input_fn": lambda x: x,
    },
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


def load_single_files_dataset(
    image_files, mask_files, relabel=False, channel_order="YXC"
):
    """Load dataset from explicit file list."""
    existing_images = [f for f in image_files if Path(f).exists()]
    existing_masks = [f for f in mask_files if Path(f).exists()]

    if not existing_images:
        return None, None, None

    images = []
    for f in tqdm(existing_images, desc="Loading images"):
        img = imread(f)
        if channel_order == "CYX" and img.ndim == 3:
            img = np.moveaxis(img, 0, -1)
        images.append(img)

    masks = []
    for f in tqdm(existing_masks, desc="Loading masks"):
        mask = imread(f)
        if relabel:
            mask = label_skimage(mask)
        masks.append(fill_label_holes(mask))

    filenames = [Path(f).name for f in existing_images]
    return images, masks, filenames


def load_ome_tiff_dataset(config):
    """Load dataset from OME-TIFF file with multiple timepoints using metadata.yml."""
    try:
        from aicsimageio import AICSImage
        import yaml
    except ImportError:
        print("  AICSImage or yaml not available for OME-TIFF loading")
        return None, None, None

    data_dir = Path(config["data_dir"])
    image_path = data_dir / config["image_file"]
    mask_path = data_dir / config["mask_file"]
    metadata_path = data_dir / config.get("metadata_file", "metadata.yml")

    if not image_path.exists() or not mask_path.exists():
        print(f"  File not found: {image_path} or {mask_path}")
        return None, None, None

    # Load metadata for channel info
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    channels = metadata.get("channels", {})
    cyan_ch = int(channels.get("cyan", 1))
    magenta_ch = int(channels.get("magenta", 0))

    img_stream = AICSImage(str(image_path))
    label_stream = AICSImage(str(mask_path))

    images = []
    masks = []
    filenames = []

    relabel = config.get("relabel", False)

    for t in tqdm(range(img_stream.dims.T), desc="Loading OME-TIFF timepoints"):
        img_cyan = img_stream.get_image_data("YX", C=cyan_ch, T=t)
        img_magenta = img_stream.get_image_data("YX", C=magenta_ch, T=t)
        # Stack channels to YXC format (only nuclear channels: cyan, magenta)
        img = np.stack([img_cyan, img_magenta], axis=-1)
        images.append(img)

        gt_labels = label_stream.get_image_data("YX", Z=t)
        if relabel:
            masks.append(fill_label_holes(label_skimage(gt_labels)))
        else:
            masks.append(gt_labels)

        filenames.append(f"t{t:03d}.tif")

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
    elif dataset_type == "single_files":
        return load_single_files_dataset(
            config["image_files"],
            config["mask_files"],
            relabel=relabel,
            channel_order=channel_order,
        )
    elif dataset_type == "ome_tiff":
        return load_ome_tiff_dataset(config)
    return None, None, None


# =============================================================================
# SNR and Accuracy Analysis Functions
# =============================================================================


def compute_snr_per_nucleus(images, masks):
    """
    Compute SNR for each nucleus, return dict mapping (img_idx, label_id) to SNR.

    Note: For multichannel images, only nuclear channels (0: cyan, 1: magenta) are used.
    The tubulin channel (2) is cytoplasmic and excluded from SNR computation.
    """
    snr_data = {}

    for img_idx, (img, mask) in enumerate(
        tqdm(zip(images, masks), total=len(images), desc="Computing SNR")
    ):
        # Compute background intensity once for the entire image
        # (only for nuclear channels - tubulin is excluded by default)
        background_intensity = compute_background_intensity(img, mask)
        if background_intensity is None:
            continue

        labels = np.unique(mask)
        labels = labels[labels > 0]

        for label_id in labels:
            # SNR computed only for nuclear channels (tubulin excluded by default)
            result = compute_snr_for_label(img, mask, label_id, background_intensity)
            if result is not None:
                snr = result["snr"]
                if isinstance(snr, list):
                    snr = max(snr)  # Max across nuclear channels only
                snr_data[(img_idx, int(label_id))] = snr

    return snr_data


def analyze_accuracy_vs_snr(images, masks, predictions, snr_data, iou_threshold=0.5):
    """
    Analyze detection accuracy as a function of SNR.

    Returns per-nucleus results (for SNR-binned recall) and overall counts
    for computing accuracy = TP / (TP + FN + FP).
    """
    nucleus_results = []
    total_tp = 0
    total_fn = 0
    total_fp = 0

    for img_idx, (mask_gt, mask_pred) in enumerate(zip(masks, predictions)):
        gt_labels = np.unique(mask_gt)
        gt_labels = gt_labels[gt_labels > 0]
        pred_labels_all = np.unique(mask_pred)
        pred_labels_all = pred_labels_all[pred_labels_all > 0]

        # Track which predictions matched a GT
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

            # Only add to SNR-binned results if we have SNR data
            if snr is not None:
                nucleus_results.append((snr, detected))

        # Count false positives (predictions that didn't match any GT)
        for pred_label in pred_labels_all:
            if pred_label not in matched_predictions:
                total_fp += 1

    # Compute overall metrics
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


def run_predictions(model, images_norm, input_fn, scale=None):
    """Run model predictions on normalized images."""
    predictions = []
    for x in tqdm(images_norm, desc="Predicting"):
        pred_input = input_fn(x)
        if scale is not None:
            pred_mask, _ = model.predict_instances(pred_input, scale=scale)
        else:
            pred_mask, _ = model.predict_instances(pred_input)
        predictions.append(pred_mask)
    return predictions


# =============================================================================
# Main
# =============================================================================


def main():
    # Check GPU
    use_gpu = gputools_available()
    print(f"Using GPU: {use_gpu}")

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        limit_gpu_memory(0.5, total_memory=50000)

    # Collect all datasets
    all_datasets = {}

    # Load validation data
    print("\nLoading validation data...")
    if Path(f"{TRAINING_DATA_DIR}/dataset_split.json").exists():
        X_val, Y_val, _ = load_dataset_with_split(TRAINING_DATA_DIR, "validation")
        all_datasets["Validation"] = {"images": X_val, "masks": Y_val, "scale": None}
    else:
        print(f"Warning: {TRAINING_DATA_DIR}/dataset_split.json not found")

    # Load external/test datasets
    print("\nLoading external datasets...")
    for name, config in EXTERNAL_DATASETS.items():
        print(f"  Trying to load {name}...")
        try:
            X_ext, Y_ext, _ = load_external_dataset(config)
            if X_ext is not None and len(X_ext) > 0:
                all_datasets[name] = {
                    "images": X_ext,
                    "masks": Y_ext,
                    "scale": config.get("scale", None),
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

    # Load models
    models = {}
    for model_name, config in MODEL_CONFIGS.items():
        model_path = Path(config["basedir"])
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping {model_name}")
            continue
        try:
            models[model_name] = {
                "model": StarDist2D(
                    None, name=config["name"], basedir=config["basedir"]
                ),
                "input_fn": config["input_fn"],
            }
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    if not models:
        print("No models found.")
        return

    # Process each dataset
    all_results = {}

    for dataset_name, dataset_info in all_datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 60}")

        images = dataset_info["images"]
        masks = dataset_info["masks"]
        scale = dataset_info["scale"]

        # Normalize images
        print("Normalizing images...")
        images_norm = [normalize(x, 1, 99.8, axis=AXIS_NORM) for x in tqdm(images)]

        # Compute SNR
        print("Computing SNR...")
        snr_data = compute_snr_per_nucleus(images, masks)
        print(f"  Computed SNR for {len(snr_data)} nuclei")

        dataset_results = {}

        # Determine number of channels in this dataset
        n_channels = images[0].shape[-1] if images[0].ndim == 3 else 1

        for model_name, model_info in models.items():
            # Skip models that require channels not present in the dataset
            if model_name == "1 CH (tubulin)" and n_channels < 3:
                print(f"\nSkipping {model_name} (dataset has only {n_channels} channels, tubulin not available)")
                continue
            if model_name == "3 CH (all)" and n_channels < 3:
                print(f"\nSkipping {model_name} (dataset has only {n_channels} channels)")
                continue

            print(f"\nRunning {model_name}...")
            model = model_info["model"]
            input_fn = model_info["input_fn"]

            predictions = run_predictions(model, images_norm, input_fn, scale=scale)

            nucleus_results, overall_metrics = analyze_accuracy_vs_snr(
                images, masks, predictions, snr_data, iou_threshold=IOU_THRESHOLD
            )

            bin_centers, recalls, counts, bin_edges = bin_and_compute_recall(
                nucleus_results, n_bins=SNR_BINS
            )

            print(f"  TP={overall_metrics['tp']}, FN={overall_metrics['fn']}, FP={overall_metrics['fp']}")
            print(f"  Recall: {overall_metrics['recall']:.3f}, Precision: {overall_metrics['precision']:.3f}")
            print(f"  Accuracy (Jaccard): {overall_metrics['accuracy']:.3f}")

            dataset_results[model_name] = {
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
            }

        all_results[dataset_name] = dataset_results

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # Colors and markers
    model_colors = {
        "1 CH (tubulin)": "blue",
        "2 CH (cyan+magenta)": "orange",
        "3 CH (all)": "green",
    }
    model_markers = {
        "1 CH (tubulin)": "o",
        "2 CH (cyan+magenta)": "s",
        "3 CH (all)": "^",
    }

    # Plot 1: Detection rate vs SNR for each dataset
    n_datasets = len(all_results)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )

    for idx, (dataset_name, dataset_results) in enumerate(all_results.items()):
        ax = axes[idx // n_cols, idx % n_cols]

        for model_name, data in dataset_results.items():
            if data["bin_centers"]:
                ax.plot(
                    data["bin_centers"],
                    data["recalls"],
                    marker=model_markers.get(model_name, "o"),
                    color=model_colors.get(model_name, "gray"),
                    label=f"{model_name} (R={data['overall_recall']:.2f})",
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("SNR")
        ax.set_ylabel("Detection Rate")
        ax.set_title(f"{dataset_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Hide empty subplots
    for idx in range(n_datasets, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig("snr_accuracy_by_dataset.pdf")
    plt.savefig("snr_accuracy_by_dataset.png", dpi=150)
    print("Saved: snr_accuracy_by_dataset.pdf/png")
    plt.show()

    # Plot 2: Summary comparison across datasets (3 CH model only)
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (dataset_name, dataset_results) in enumerate(all_results.items()):
        if "3 CH (all)" in dataset_results:
            data = dataset_results["3 CH (all)"]
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
    ax.set_title(f"Detection Rate vs SNR - 3 Channel Model (IoU={IOU_THRESHOLD})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("snr_accuracy_comparison.pdf")
    plt.savefig("snr_accuracy_comparison.png", dpi=150)
    print("Saved: snr_accuracy_comparison.pdf/png")
    plt.show()

    # Save results to JSON
    output_data = {}
    for dataset_name, dataset_results in all_results.items():
        output_data[dataset_name] = {}
        for model_name, data in dataset_results.items():
            snr_vals = [r[0] for r in data["nucleus_results"]]
            output_data[dataset_name][model_name] = {
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
                "snr_mean": float(np.mean(snr_vals)) if snr_vals else None,
                "snr_std": float(np.std(snr_vals)) if snr_vals else None,
            }

    with open("snr_accuracy_results.json", "w") as fp:
        json.dump(output_data, fp, indent=2)
    print("Saved: snr_accuracy_results.json")

    # Print summary tables
    print("\n" + "=" * 100)
    print("Summary: Detection Metrics by Dataset and Model")
    print("=" * 100)

    # Table 1: Recall
    print("\nRecall = TP / (TP + FN)")
    print(f"{'Dataset':<20} {'1 CH':>12} {'2 CH':>12} {'3 CH':>12}")
    print("-" * 60)
    for dataset_name, dataset_results in all_results.items():
        vals = []
        for model_key in ["1 CH (tubulin)", "2 CH (cyan+magenta)", "3 CH (all)"]:
            if model_key in dataset_results:
                vals.append(f"{dataset_results[model_key]['overall_recall']:.3f}")
            else:
                vals.append("N/A")
        print(f"{dataset_name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Table 2: Precision
    print("\nPrecision = TP / (TP + FP)")
    print(f"{'Dataset':<20} {'1 CH':>12} {'2 CH':>12} {'3 CH':>12}")
    print("-" * 60)
    for dataset_name, dataset_results in all_results.items():
        vals = []
        for model_key in ["1 CH (tubulin)", "2 CH (cyan+magenta)", "3 CH (all)"]:
            if model_key in dataset_results:
                vals.append(f"{dataset_results[model_key]['overall_precision']:.3f}")
            else:
                vals.append("N/A")
        print(f"{dataset_name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Table 3: Accuracy (Jaccard)
    print("\nAccuracy (Jaccard) = TP / (TP + FN + FP)")
    print(f"{'Dataset':<20} {'1 CH':>12} {'2 CH':>12} {'3 CH':>12}")
    print("-" * 60)
    for dataset_name, dataset_results in all_results.items():
        vals = []
        for model_key in ["1 CH (tubulin)", "2 CH (cyan+magenta)", "3 CH (all)"]:
            if model_key in dataset_results:
                vals.append(f"{dataset_results[model_key]['overall_accuracy']:.3f}")
            else:
                vals.append("N/A")
        print(f"{dataset_name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("=" * 100)


if __name__ == "__main__":
    main()
