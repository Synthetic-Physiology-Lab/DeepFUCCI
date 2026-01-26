"""
Generate histograms of Signal-to-Noise Ratio (SNR) for nuclei across different datasets.

This script computes SNR for each nucleus in training, validation, test, and external
datasets using the ground truth segmentation masks.

SNR = (I_in - I_out) / std_in
Reference: https://imagej.net/plugins/trackmate/analyzers/#contrast--signalnoise-ratio

IMPORTANT: For FUCCI data with 3 channels, only the nuclear channels are used:
- Channel 0: Cyan (G1 phase marker) - NUCLEAR
- Channel 1: Magenta (S/G2/M phase marker) - NUCLEAR
- Channel 2: Tubulin - CYTOPLASMIC (EXCLUDED from SNR analysis)

The tubulin channel is cytoplasmic and has no nuclear signal, so including it
would distort the SNR analysis.

Usage:
    python snr_histogram.py

Requires:
    - training_data/ folder with images/, masks/, and dataset_split.json
    - test_data_tiled/ folder with images/ and masks/
    - External dataset folders (optional, configure in EXTERNAL_DATASETS)
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.measure import label as label_skimage
from stardist import fill_label_holes
from tqdm import tqdm

from snr_utils import compute_snr_for_dataset

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "../../data"
TRAINING_DATA_DIR = f"{DATA_DIR}/training_data_tiled_strict_classified"
AXIS_NORM = (0, 1)  # Normalize channels independently

# External test datasets configuration
# Each entry defines how to load images and masks for an external dataset
# Format:
#   "name": {
#       "type": "directory" | "single_files",
#       "images_dir": path to images directory (for "directory" type),
#       "masks_dir": path to masks directory (for "directory" type),
#       "image_files": list of image file paths (for "single_files" type),
#       "mask_files": list of mask file paths (for "single_files" type),
#       "relabel": True if masks need skimage.label() (binary masks),
#       "channel_order": "CYX" or "YXC" for channel axis position,
#   }

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
    },
    "CottonEtAl": {
        "type": "single_files",
        "image_files": [f"{DATA_DIR}/test_cottonetal/frame_69.tif"],
        "mask_files": [f"{DATA_DIR}/test_cottonetal/gt_frame_69.tif"],
        "relabel": True,
        "channel_order": "CYX",
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


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_dataset_with_split(data_dir, split_key):
    """Load images and masks for a specific split (training/validation)."""
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
    """Load dataset from directory with images/ and masks/ subfolders."""
    image_files = sorted(glob(f"{images_dir}/*.tif"))
    mask_files = sorted(glob(f"{masks_dir}/*.tif"))

    if not image_files:
        return None, None, None

    assert len(image_files) == len(mask_files), (
        f"Mismatch: {len(image_files)} images vs {len(mask_files)} masks"
    )
    assert all(Path(x).name == Path(y).name for x, y in zip(image_files, mask_files)), (
        "Image and mask filenames don't match"
    )

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
    """Load dataset from explicit list of image and mask files."""
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
    """Load an external dataset based on its configuration."""
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
    else:
        print(f"Unknown dataset type: {dataset_type}")
        return None, None, None


# =============================================================================
# Main
# =============================================================================


def main():
    datasets = {}

    # Load training data
    print("Loading training data...")
    if Path(f"{TRAINING_DATA_DIR}/dataset_split.json").exists():
        X_train, Y_train, _ = load_dataset_with_split(TRAINING_DATA_DIR, "training")
        datasets["Training"] = (X_train, Y_train)

        # Load validation data
        print("Loading validation data...")
        X_val, Y_val, _ = load_dataset_with_split(TRAINING_DATA_DIR, "validation")
        datasets["Validation"] = (X_val, Y_val)
    else:
        print(
            f"Warning: {TRAINING_DATA_DIR}/dataset_split.json not found. "
            "Skipping training/validation data."
        )

    # Load external/test datasets
    print("\nLoading external datasets...")
    for name, config in EXTERNAL_DATASETS.items():
        print(f"  Trying to load {name}...")
        try:
            X_ext, Y_ext, _ = load_external_dataset(config)
            if X_ext is not None and len(X_ext) > 0:
                datasets[name] = (X_ext, Y_ext)
                print(f"    Loaded {len(X_ext)} images")
            else:
                print("    Not found or empty, skipping")
        except Exception as e:
            print(f"    Error loading {name}: {e}")

    if not datasets:
        print("No datasets found. Please check the data directories.")
        return

    print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Compute SNR for each dataset
    snr_results = {}
    for name, (images, masks) in datasets.items():
        print(f"\nComputing SNR for {name} dataset ({len(images)} images)...")
        result = compute_snr_for_dataset(images, masks)
        snr_results[name] = result
        print(f"  Found {len(result['all_snr'])} nuclei")
        if result["all_snr"]:
            print(
                f"  SNR range: [{min(result['all_snr']):.2f}, {max(result['all_snr']):.2f}]"
            )
            print(
                f"  SNR mean: {np.mean(result['all_snr']):.2f}, "
                f"median: {np.median(result['all_snr']):.2f}"
            )

    # Define colors for datasets
    base_colors = {
        "Training": "blue",
        "Validation": "orange",
        "Test": "green",
    }
    # Add colors for external datasets
    extra_colors = ["red", "purple", "brown", "pink", "olive", "cyan"]
    color_idx = 0
    colors = base_colors.copy()
    for name in snr_results.keys():
        if name not in colors:
            colors[name] = extra_colors[color_idx % len(extra_colors)]
            color_idx += 1

    # Plot combined histogram
    n_datasets = len(snr_results)
    fig_width = max(14, 7 + n_datasets)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))

    # Histogram
    ax1 = axes[0]
    for name, result in snr_results.items():
        snr_vals = np.array(result["all_snr"])
        ax1.hist(
            snr_vals,
            bins=50,
            alpha=0.5,
            label=f"{name} (n={len(snr_vals)})",
            color=colors.get(name, "gray"),
            density=True,
        )

    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("Density")
    ax1.set_title("SNR Distribution Across Datasets")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    box_data = [np.array(result["all_snr"]) for result in snr_results.values()]
    box_labels = list(snr_results.keys())
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

    for patch, name in zip(bp["boxes"], box_labels):
        patch.set_facecolor(colors.get(name, "gray"))
        patch.set_alpha(0.5)

    ax2.set_ylabel("Signal-to-Noise Ratio (SNR)")
    ax2.set_title("SNR Distribution by Dataset")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("snr_histogram_all_datasets.pdf")
    plt.savefig("snr_histogram_all_datasets.png", dpi=150)
    print("\nSaved: snr_histogram_all_datasets.pdf/png")
    plt.show()

    # Per-channel histograms if multichannel
    has_channels = any(
        result["channel_snr"] is not None for result in snr_results.values()
    )

    if has_channels:
        # Only nuclear channels are used (tubulin is cytoplasmic, excluded)
        channel_names = ["Cyan (G1)", "Magenta (S/G2/M)"]
        n_channels = max(
            len(result["channel_snr"])
            for result in snr_results.values()
            if result["channel_snr"] is not None
        )

        fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 4))
        if n_channels == 1:
            axes = [axes]

        for c in range(n_channels):
            ax = axes[c]
            for name, result in snr_results.items():
                if result["channel_snr"] is not None and c in result["channel_snr"]:
                    snr_vals = np.array(result["channel_snr"][c])
                    ax.hist(
                        snr_vals,
                        bins=50,
                        alpha=0.5,
                        label=f"{name} (n={len(snr_vals)})",
                        color=colors.get(name, "gray"),
                        density=True,
                    )

            channel_label = (
                channel_names[c] if c < len(channel_names) else f"Channel {c}"
            )
            ax.set_xlabel("SNR")
            ax.set_ylabel("Density")
            ax.set_title(f"SNR Distribution - {channel_label}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("snr_histogram_per_channel.pdf")
        plt.savefig("snr_histogram_per_channel.png", dpi=150)
        print("Saved: snr_histogram_per_channel.pdf/png")
        plt.show()

    # Save SNR data to JSON for further analysis
    output_data = {}
    for name, result in snr_results.items():
        output_data[name] = {
            "all_snr": result["all_snr"],
            "n_nuclei": len(result["all_snr"]),
            "mean": float(np.mean(result["all_snr"])) if result["all_snr"] else None,
            "median": float(np.median(result["all_snr"]))
            if result["all_snr"]
            else None,
            "std": float(np.std(result["all_snr"])) if result["all_snr"] else None,
            "min": float(min(result["all_snr"])) if result["all_snr"] else None,
            "max": float(max(result["all_snr"])) if result["all_snr"] else None,
        }

    with open("snr_statistics.json", "w") as fp:
        json.dump(output_data, fp, indent=2)
    print("Saved: snr_statistics.json")

    # Print summary table
    print("\n" + "=" * 70)
    print("SNR Statistics Summary")
    print("=" * 70)
    print(f"{'Dataset':<20} {'N nuclei':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print("-" * 70)
    for name, data in output_data.items():
        print(
            f"{name:<20} {data['n_nuclei']:>10} "
            f"{data['mean']:>10.2f} {data['median']:>10.2f} {data['std']:>10.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
