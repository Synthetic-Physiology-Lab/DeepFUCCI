"""
Analyze SNR for model predictions vs ground truth.

This script computes the Signal-to-Noise Ratio (SNR) for nuclei that are
correctly detected (IoU >= 0.5) vs missed (IoU < 0.5) by segmentation models.

The analysis tests whether model failures are driven by low SNR (hard to see
nuclei) or by other factors.

Models analyzed:
- DAPIeq raw: DAPI-equivalent without postprocessing
- DAPIeq post: DAPI-equivalent with top-hat and gaussian blur
- StarDist 2CH: Custom-trained 2-channel model (for comparison)

IMPORTANT: For FUCCI data with 3 channels, only the nuclear channels are used for SNR:
- Channel 0: Cyan (G1 phase marker) - NUCLEAR
- Channel 1: Magenta (S/G2/M phase marker) - NUCLEAR
- Channel 2: Tubulin - CYTOPLASMIC (EXCLUDED from SNR analysis)

Usage:
    python analyze_model_snr.py

Output:
    - model_snr_analysis.pdf/png: SNR distribution plots
    - model_snr_results.json: Detailed statistics
"""

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from csbdeep.utils import normalize
from scipy import stats
from skimage.io import imread
from skimage.measure import label as label_skimage
from stardist import fill_label_holes
from stardist.models import StarDist2D
from tqdm import tqdm

from snr_utils import compute_background_intensity, compute_snr_for_label

try:
    import pyclesperanto_prototype as cle
    HAS_PYCLESPERANTO = True
except ImportError:
    HAS_PYCLESPERANTO = False
    print("Warning: pyclesperanto not available, DAPI-eq postprocessing disabled")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path.home() / "models"
IOU_THRESHOLD = 0.5
NUCLEUS_RADIUS_PIXEL = 10 / 0.3  # 10 microns / 0.3 microns per pixel

DATASETS = {
    "HT1080_20x": DATA_DIR / "data_set_HT1080_20x",
    "HT1080_40x": DATA_DIR / "data_set_HT1080_40x",
}


# =============================================================================
# Prediction Functions
# =============================================================================


def dapieq_predict_raw(model, img):
    """DAPI-equivalent prediction without postprocessing."""
    ch1 = img[..., 0]
    ch2 = img[..., 1]
    max_projected = np.maximum(ch1, ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


def dapieq_predict_postprocessed(model, img):
    """DAPI-equivalent prediction with top-hat and gaussian blur."""
    if not HAS_PYCLESPERANTO:
        return dapieq_predict_raw(model, img)

    ch1 = img[..., 0]
    ch2 = img[..., 1]

    ch1_top = cle.top_hat_sphere(
        ch1, radius_x=2.0 * NUCLEUS_RADIUS_PIXEL, radius_y=2.0 * NUCLEUS_RADIUS_PIXEL
    )
    ch1_blur = cle.gaussian_blur(ch1_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch1 = normalize(ch1_blur.get())

    ch2_top = cle.top_hat_sphere(
        ch2, radius_x=2.0 * NUCLEUS_RADIUS_PIXEL, radius_y=2.0 * NUCLEUS_RADIUS_PIXEL
    )
    ch2_blur = cle.gaussian_blur(ch2_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch2 = normalize(ch2_blur.get())

    max_projected = np.maximum(normal_ch1, normal_ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


def stardist_2ch_predict(model, img):
    """StarDist 2-channel prediction."""
    labels, _ = model.predict_instances(img[..., 0:2])
    return labels


# =============================================================================
# Helper Functions
# =============================================================================


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union if union > 0 else 0.0


def find_best_match_iou(gt_label_mask, pred_mask):
    """
    Find the best matching prediction for a given GT label.
    Returns the maximum IoU achieved with any overlapping prediction.
    """
    overlapping_preds = np.unique(pred_mask[gt_label_mask])
    overlapping_preds = overlapping_preds[overlapping_preds > 0]

    if len(overlapping_preds) == 0:
        return 0.0

    best_iou = 0.0
    for pred_label in overlapping_preds:
        pred_label_mask = pred_mask == pred_label
        iou = compute_iou(gt_label_mask, pred_label_mask)
        best_iou = max(best_iou, iou)

    return best_iou


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_model_on_dataset(model, predict_fn, dataset_name, dataset_path):
    """
    Analyze SNR for detected vs missed nuclei for a model on a dataset.

    We analyze from the GT perspective: for each GT nucleus, check if the model
    detected it (IoU >= 0.5 with any prediction) and compute its SNR.
    """
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"

    image_files = sorted(glob(str(images_dir / "*.tif")))
    mask_files = sorted(glob(str(masks_dir / "*.tif")))

    # Match by filename
    mask_names = {Path(f).name for f in mask_files}
    matched = [(f, str(masks_dir / Path(f).name)) for f in image_files
               if Path(f).name in mask_names]

    detected_snr = []
    missed_snr = []

    for img_path, mask_path in tqdm(matched, desc=f"    {dataset_name}", leave=False):
        # Load and normalize image
        image_raw = imread(img_path)
        image = normalize(image_raw, 1, 99.8, axis=(0, 1))

        # Load GT mask
        gt_mask = imread(mask_path)
        gt_mask = fill_label_holes(label_skimage(gt_mask))

        # Get model prediction
        pred_mask = predict_fn(model, image)

        # Compute background intensity for SNR (using GT mask for background)
        background_intensity = compute_background_intensity(image_raw, gt_mask)
        if background_intensity is None:
            continue

        # For each GT nucleus, check if detected and compute SNR
        gt_labels = np.unique(gt_mask)
        gt_labels = gt_labels[gt_labels > 0]

        for label_id in gt_labels:
            gt_label_mask = gt_mask == label_id
            best_iou = find_best_match_iou(gt_label_mask, pred_mask)

            # Compute SNR for this GT nucleus
            snr_result = compute_snr_for_label(
                image_raw, gt_mask, label_id, background_intensity
            )
            if snr_result is None:
                continue

            snr = snr_result["snr"]
            if isinstance(snr, list):
                snr = max(snr)  # Use max across nuclear channels

            if best_iou >= IOU_THRESHOLD:
                detected_snr.append(snr)
            else:
                missed_snr.append(snr)

    return {
        "detected_snr": detected_snr,
        "missed_snr": missed_snr,
        "n_detected": len(detected_snr),
        "n_missed": len(missed_snr),
    }


def main():
    print("=" * 70)
    print("SNR Analysis for Model Predictions vs Ground Truth")
    print("=" * 70)
    print()
    print("This analyzes whether model failures (missed nuclei) correlate with low SNR.")
    print()

    # Load models
    print("Loading models...")

    # Pretrained model for DAPI-eq
    dapieq_model = StarDist2D.from_pretrained("2D_versatile_fluo")
    print("  Loaded: 2D_versatile_fluo (for DAPI-eq)")

    # Custom 2-channel model
    stardist_2ch = StarDist2D(None, name="stardist_2_channel_latest", basedir=MODEL_DIR)
    print("  Loaded: StarDist 2CH")

    # Define models to analyze
    models = {
        "dapieq_raw": {
            "model": dapieq_model,
            "predict_fn": dapieq_predict_raw,
            "display_name": "DAPIeq raw",
        },
        "dapieq_post": {
            "model": dapieq_model,
            "predict_fn": dapieq_predict_postprocessed,
            "display_name": "DAPIeq post",
        },
        "stardist_2ch": {
            "model": stardist_2ch,
            "predict_fn": stardist_2ch_predict,
            "display_name": "StarDist 2CH",
        },
    }

    # Results storage
    results = {
        "iou_threshold": IOU_THRESHOLD,
        "models": {},
    }

    # Analyze each model
    for model_key, model_info in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_info['display_name']}")
        print(f"{'='*70}")

        model_results = {
            "display_name": model_info["display_name"],
            "per_dataset": {},
            "overall": {},
        }

        all_detected_snr = []
        all_missed_snr = []

        for dataset_name, dataset_path in DATASETS.items():
            print(f"\n  Dataset: {dataset_name}")

            dataset_result = analyze_model_on_dataset(
                model_info["model"],
                model_info["predict_fn"],
                dataset_name,
                dataset_path,
            )

            model_results["per_dataset"][dataset_name] = {
                "n_detected": dataset_result["n_detected"],
                "n_missed": dataset_result["n_missed"],
                "mean_snr_detected": float(np.mean(dataset_result["detected_snr"])) if dataset_result["detected_snr"] else None,
                "mean_snr_missed": float(np.mean(dataset_result["missed_snr"])) if dataset_result["missed_snr"] else None,
            }

            all_detected_snr.extend(dataset_result["detected_snr"])
            all_missed_snr.extend(dataset_result["missed_snr"])

            # Print dataset summary
            det = dataset_result["detected_snr"]
            mis = dataset_result["missed_snr"]
            recall = len(det) / (len(det) + len(mis)) * 100 if (det or mis) else 0
            print(f"    Detected: n={len(det)}, mean SNR={np.mean(det):.2f}" if det else f"    Detected: n=0")
            print(f"    Missed: n={len(mis)}, mean SNR={np.mean(mis):.2f}" if mis else f"    Missed: n=0")
            print(f"    Recall: {recall:.1f}%")

        # Overall statistics
        if all_detected_snr and all_missed_snr:
            t_stat, p_ttest = stats.ttest_ind(all_detected_snr, all_missed_snr)
            u_stat, p_mw = stats.mannwhitneyu(
                all_detected_snr, all_missed_snr, alternative="two-sided"
            )

            model_results["overall"] = {
                "detected": {
                    "n": len(all_detected_snr),
                    "mean_snr": float(np.mean(all_detected_snr)),
                    "median_snr": float(np.median(all_detected_snr)),
                    "std_snr": float(np.std(all_detected_snr)),
                },
                "missed": {
                    "n": len(all_missed_snr),
                    "mean_snr": float(np.mean(all_missed_snr)),
                    "median_snr": float(np.median(all_missed_snr)),
                    "std_snr": float(np.std(all_missed_snr)),
                },
                "statistics": {
                    "ttest_t": float(t_stat),
                    "ttest_p": float(p_ttest),
                    "mannwhitney_u": float(u_stat),
                    "mannwhitney_p": float(p_mw),
                },
            }

            print(f"\n  Overall for {model_info['display_name']}:")
            print(f"    Detected: n={len(all_detected_snr)}, mean SNR={np.mean(all_detected_snr):.3f}")
            print(f"    Missed: n={len(all_missed_snr)}, mean SNR={np.mean(all_missed_snr):.3f}")
            print(f"    SNR difference: {np.mean(all_detected_snr) - np.mean(all_missed_snr):+.3f}")
            print(f"    Mann-Whitney p-value: {p_mw:.2e}")

        results["models"][model_key] = model_results

    # Generate comparison plot
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (model_key, model_info) in enumerate(models.items()):
        ax = axes[idx]
        model_data = results["models"][model_key]

        if "overall" not in model_data or not model_data["overall"]:
            continue

        # Collect SNR data for this model
        detected_snr = []
        missed_snr = []
        for dataset_name in DATASETS.keys():
            dataset_result = analyze_model_on_dataset(
                model_info["model"],
                model_info["predict_fn"],
                dataset_name,
                DATASETS[dataset_name],
            )
            detected_snr.extend(dataset_result["detected_snr"])
            missed_snr.extend(dataset_result["missed_snr"])

        # Box plot
        data = [detected_snr, missed_snr]
        bp = ax.boxplot(data, labels=["Detected", "Missed"], patch_artist=True)
        bp["boxes"][0].set_facecolor("C0")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("C1")
        bp["boxes"][1].set_alpha(0.6)

        p_val = model_data["overall"]["statistics"]["mannwhitney_p"]
        ax.set_title(f"{model_info['display_name']}\n(p={p_val:.2e})")
        ax.set_ylabel("SNR")
        ax.grid(True, alpha=0.3, axis="y")

        # Add mean lines
        ax.axhline(np.mean(detected_snr), color="C0", linestyle="--", alpha=0.7)
        ax.axhline(np.mean(missed_snr), color="C1", linestyle="--", alpha=0.7)

    plt.suptitle("SNR of Detected vs Missed Nuclei by Model", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("model_snr_analysis.pdf", bbox_inches="tight")
    plt.savefig("model_snr_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved: model_snr_analysis.pdf/png")
    plt.close()

    # Save results
    with open("model_snr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: model_snr_results.json")

    # Print summary table
    print(f"\n{'='*70}")
    print("Summary: SNR of Detected vs Missed Nuclei")
    print(f"{'='*70}")
    print()
    print(f"{'Model':<15} {'Detected SNR':>15} {'Missed SNR':>15} {'Difference':>12} {'p-value':>12}")
    print("-" * 70)

    for model_key, model_data in results["models"].items():
        if "overall" not in model_data or not model_data["overall"]:
            continue

        display_name = model_data["display_name"]
        det_snr = model_data["overall"]["detected"]["mean_snr"]
        mis_snr = model_data["overall"]["missed"]["mean_snr"]
        diff = det_snr - mis_snr
        p_val = model_data["overall"]["statistics"]["mannwhitney_p"]

        print(f"{display_name:<15} {det_snr:>15.3f} {mis_snr:>15.3f} {diff:>+12.3f} {p_val:>12.2e}")

    print()
    print("Interpretation:")
    print("  - Higher SNR for detected nuclei suggests low SNR drives missed detections")
    print("  - Similar SNR suggests other factors (shape, overlap, etc.) drive errors")


if __name__ == "__main__":
    main()
