"""
Validate segmentation models against independent annotator masks.

This script validates all segmentation methods against both:
1. Primary annotator masks (masks/) - original ground truth
2. Independent annotator masks (masks_independent/) - secondary annotation

This tests whether model performance is robust across different annotators
or whether models simply learned the primary annotator's annotation style.

Usage:
    python validate_against_independent.py

Output:
    - validation_independent_results.json: Accuracy results for all methods
    - validation_independent_comparison.csv: Side-by-side comparison table
"""

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from skimage.io import imread
from skimage.measure import label as label_skimage
from stardist import fill_label_holes
from stardist.matching import matching_dataset
from stardist.models import StarDist2D
from tqdm import tqdm

try:
    import pyclesperanto_prototype as cle
    HAS_PYCLESPERANTO = True
except ImportError:
    HAS_PYCLESPERANTO = False
    print("Warning: pyclesperanto not available, DAPI-eq postprocessing disabled")

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODEL_DIR = Path.home() / "models"
IOU_THRESHOLD = 0.5
NUCLEUS_RADIUS_PIXEL = 10 / 0.3  # 10 microns divided by 0.3 microns per pixel

DATASETS = {
    "ht1080_20x": {
        "path": DATA_DIR / "data_set_HT1080_20x",
        "display_name": "HT1080 20x",
    },
    "ht1080_40x": {
        "path": DATA_DIR / "data_set_HT1080_40x",
        "display_name": "HT1080 40x",
    },
}

# Model configurations for custom-trained StarDist models
STARDIST_MODELS = {
    "stardist_1ch": {
        "name": "stardist_1_channel_latest",
        "channels": lambda x: x[..., 2],  # Tubulin channel
        "display_name": "StarDist 1CH",
    },
    "stardist_2ch": {
        "name": "stardist_2_channel_latest",
        "channels": lambda x: x[..., 0:2],  # Cyan + Magenta
        "display_name": "StarDist 2CH",
    },
    "stardist_3ch": {
        "name": "stardist_3_channel_latest",
        "channels": lambda x: x,  # All channels
        "display_name": "StarDist 3CH",
    },
}

# DAPI-equivalent method configurations
DAPIEQ_METHODS = {
    "dapieq_raw": {
        "display_name": "DAPIeq raw",
        "postprocess": False,
    },
    "dapieq_post": {
        "display_name": "DAPIeq post",
        "postprocess": True,
    },
}


def dapieq_predict_raw(model, img):
    """
    DAPI-equivalent prediction without postprocessing.
    Takes max projection of cyan and magenta channels.
    """
    ch1 = img[..., 0]
    ch2 = img[..., 1]
    max_projected = np.maximum(ch1, ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


def dapieq_predict_postprocessed(model, img):
    """
    DAPI-equivalent prediction with top-hat and gaussian blur postprocessing.
    """
    if not HAS_PYCLESPERANTO:
        return dapieq_predict_raw(model, img)

    ch1 = img[..., 0]
    ch2 = img[..., 1]

    # Top-hat filtering and gaussian blur for channel 1
    ch1_top = cle.top_hat_sphere(
        ch1, radius_x=2.0 * NUCLEUS_RADIUS_PIXEL, radius_y=2.0 * NUCLEUS_RADIUS_PIXEL
    )
    ch1_blur = cle.gaussian_blur(ch1_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch1 = normalize(ch1_blur.get())

    # Top-hat filtering and gaussian blur for channel 2
    ch2_top = cle.top_hat_sphere(
        ch2, radius_x=2.0 * NUCLEUS_RADIUS_PIXEL, radius_y=2.0 * NUCLEUS_RADIUS_PIXEL
    )
    ch2_blur = cle.gaussian_blur(ch2_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch2 = normalize(ch2_blur.get())

    max_projected = np.maximum(normal_ch1, normal_ch2)
    labels, _ = model.predict_instances(max_projected)
    return labels


def load_images_and_masks(dataset_path, mask_folder="masks"):
    """
    Load images and masks from a dataset directory.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.
    mask_folder : str
        Name of the mask folder ('masks' or 'masks_independent').

    Returns
    -------
    tuple
        (images, masks, filenames) - normalized images, filled masks, and filenames
    """
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / mask_folder

    if not masks_dir.exists():
        print(f"  Warning: {masks_dir} not found")
        return [], [], []

    image_files = sorted(glob(str(images_dir / "*.tif")))
    mask_files = sorted(glob(str(masks_dir / "*.tif")))

    # Match files by name
    mask_names = {Path(f).name for f in mask_files}
    matched_images = []
    matched_masks = []
    filenames = []

    for img_path in image_files:
        name = Path(img_path).name
        if name in mask_names:
            matched_images.append(img_path)
            matched_masks.append(str(masks_dir / name))
            filenames.append(name)

    # Load and normalize
    images = [imread(f) for f in matched_images]
    masks = [imread(f) for f in matched_masks]

    # Normalize images
    axis_norm = (0, 1)  # normalize channels independently
    images = [normalize(x, 1, 99.8, axis=axis_norm) for x in images]

    # Fill label holes and ensure proper labeling
    masks = [fill_label_holes(label_skimage(m)) for m in masks]

    return images, masks, filenames


def run_model_predictions(model, images, channel_selector):
    """
    Run model predictions on a set of images.

    Parameters
    ----------
    model : StarDist2D
        The StarDist model to use.
    images : list
        List of normalized images.
    channel_selector : callable
        Function to select appropriate channels from image.

    Returns
    -------
    list
        List of predicted label masks.
    """
    predictions = []
    for img in tqdm(images, desc="    Predicting", leave=False):
        input_img = channel_selector(img)
        pred, _ = model.predict_instances(input_img)
        predictions.append(pred)
    return predictions


def compute_accuracy(gt_masks, pred_masks, threshold=0.5):
    """
    Compute matching accuracy at a given IoU threshold.

    Parameters
    ----------
    gt_masks : list
        List of ground truth masks.
    pred_masks : list
        List of predicted masks.
    threshold : float
        IoU threshold for matching.

    Returns
    -------
    dict
        Matching statistics including accuracy.
    """
    stats = matching_dataset(gt_masks, pred_masks, thresh=threshold, show_progress=False)
    return {
        "accuracy": stats.accuracy,
        "f1": stats.f1,
        "precision": stats.precision,
        "recall": stats.recall,
        "n_true": stats.n_true,
        "n_pred": stats.n_pred,
        "tp": stats.tp,
        "fp": stats.fp,
        "fn": stats.fn,
    }


def main():
    print("=" * 70)
    print("Validation Against Independent Annotator Masks")
    print("=" * 70)
    print()
    print("This script validates segmentation models against both:")
    print("  1. Primary annotator masks (original ground truth)")
    print("  2. Independent annotator masks (secondary annotation)")
    print()

    # Load custom-trained StarDist models
    print("Loading custom-trained StarDist models...")
    stardist_models = {}
    for model_key, model_config in STARDIST_MODELS.items():
        try:
            stardist_models[model_key] = StarDist2D(
                None, name=model_config["name"], basedir=MODEL_DIR
            )
            print(f"  Loaded: {model_config['display_name']}")
        except Exception as e:
            print(f"  Failed to load {model_config['display_name']}: {e}")

    # Load pretrained StarDist model for DAPI-equivalent methods
    print("\nLoading pretrained StarDist model for DAPI-equivalent...")
    try:
        dapieq_model = StarDist2D.from_pretrained("2D_versatile_fluo")
        print("  Loaded: 2D_versatile_fluo (pretrained)")
    except Exception as e:
        print(f"  Failed to load pretrained model: {e}")
        dapieq_model = None

    if not stardist_models and not dapieq_model:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # Results storage
    results = {
        "iou_threshold": IOU_THRESHOLD,
        "per_dataset": {},
        "summary": {},
    }

    # Process each dataset
    for dataset_key, dataset_info in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_info['display_name']}")
        print(f"{'='*70}")

        dataset_path = dataset_info["path"]

        # Load images and both mask sets
        print("\nLoading data...")
        images, masks_primary, filenames = load_images_and_masks(
            dataset_path, "masks"
        )
        _, masks_independent, _ = load_images_and_masks(
            dataset_path, "masks_independent"
        )

        if not images:
            print(f"  No images found for {dataset_key}")
            continue

        if not masks_independent:
            print(f"  No independent masks found for {dataset_key}")
            continue

        print(f"  Loaded {len(images)} images")
        print(f"  Primary masks: {len(masks_primary)}")
        print(f"  Independent masks: {len(masks_independent)}")

        dataset_results = {
            "n_images": len(images),
            "vs_primary": {},
            "vs_independent": {},
        }

        # Run predictions for each custom-trained StarDist model
        for model_key, model in stardist_models.items():
            model_config = STARDIST_MODELS[model_key]
            print(f"\n  Model: {model_config['display_name']}")

            # Get predictions
            predictions = run_model_predictions(
                model, images, model_config["channels"]
            )

            # Compare against primary masks
            print("    Comparing vs primary annotator...")
            stats_primary = compute_accuracy(masks_primary, predictions, IOU_THRESHOLD)
            dataset_results["vs_primary"][model_key] = stats_primary
            print(f"      Accuracy: {stats_primary['accuracy']:.3f}")

            # Compare against independent masks
            print("    Comparing vs independent annotator...")
            stats_independent = compute_accuracy(
                masks_independent, predictions, IOU_THRESHOLD
            )
            dataset_results["vs_independent"][model_key] = stats_independent
            print(f"      Accuracy: {stats_independent['accuracy']:.3f}")

            # Compute difference
            diff = stats_primary["accuracy"] - stats_independent["accuracy"]
            print(f"      Difference: {diff:+.3f}")

        # Run DAPI-equivalent methods
        if dapieq_model is not None:
            for method_key, method_config in DAPIEQ_METHODS.items():
                print(f"\n  Model: {method_config['display_name']}")

                # Get predictions using appropriate method
                predictions = []
                for img in tqdm(images, desc="    Predicting", leave=False):
                    if method_config["postprocess"]:
                        pred = dapieq_predict_postprocessed(dapieq_model, img)
                    else:
                        pred = dapieq_predict_raw(dapieq_model, img)
                    predictions.append(pred)

                # Compare against primary masks
                print("    Comparing vs primary annotator...")
                stats_primary = compute_accuracy(masks_primary, predictions, IOU_THRESHOLD)
                dataset_results["vs_primary"][method_key] = stats_primary
                print(f"      Accuracy: {stats_primary['accuracy']:.3f}")

                # Compare against independent masks
                print("    Comparing vs independent annotator...")
                stats_independent = compute_accuracy(
                    masks_independent, predictions, IOU_THRESHOLD
                )
                dataset_results["vs_independent"][method_key] = stats_independent
                print(f"      Accuracy: {stats_independent['accuracy']:.3f}")

                # Compute difference
                diff = stats_primary["accuracy"] - stats_independent["accuracy"]
                print(f"      Difference: {diff:+.3f}")

        results["per_dataset"][dataset_key] = dataset_results

    # Compute summary statistics
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    # Combine all methods for summary
    all_methods = {}
    for k, v in STARDIST_MODELS.items():
        all_methods[k] = v["display_name"]
    for k, v in DAPIEQ_METHODS.items():
        all_methods[k] = v["display_name"]

    summary = {}
    for model_key, display_name in all_methods.items():
        model_summary = {
            "vs_primary": [],
            "vs_independent": [],
        }

        for dataset_key, dataset_results in results["per_dataset"].items():
            if model_key in dataset_results["vs_primary"]:
                model_summary["vs_primary"].append(
                    dataset_results["vs_primary"][model_key]["accuracy"]
                )
            if model_key in dataset_results["vs_independent"]:
                model_summary["vs_independent"].append(
                    dataset_results["vs_independent"][model_key]["accuracy"]
                )

        if model_summary["vs_primary"] and model_summary["vs_independent"]:
            avg_primary = np.mean(model_summary["vs_primary"])
            avg_independent = np.mean(model_summary["vs_independent"])
            diff = avg_primary - avg_independent

            summary[model_key] = {
                "avg_vs_primary": avg_primary,
                "avg_vs_independent": avg_independent,
                "difference": diff,
            }

            print(f"\n{display_name}:")
            print(f"  Avg accuracy vs primary:     {avg_primary:.3f}")
            print(f"  Avg accuracy vs independent: {avg_independent:.3f}")
            print(f"  Difference:                  {diff:+.3f}")

    results["summary"] = summary

    # Save results
    output_json = Path(__file__).parent / "validation_independent_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_json}")

    # Generate comparison CSV
    output_csv = Path(__file__).parent / "validation_independent_comparison.csv"
    generate_comparison_csv(results, output_csv)
    print(f"Comparison table saved to: {output_csv}")

    # Print interpretation
    print(f"\n{'='*70}")
    print("Interpretation")
    print(f"{'='*70}")
    print()
    print("If accuracy vs independent ≈ accuracy vs primary:")
    print("  → Models learned generalizable nuclear features")
    print()
    print("If accuracy vs independent < accuracy vs primary:")
    print("  → Models may have learned annotation-specific biases")
    print()
    print("If accuracy vs independent > inter-annotator agreement (~85%):")
    print("  → Models are robust across annotation styles")


def generate_comparison_csv(results, output_path):
    """Generate a CSV table comparing results."""
    lines = ["Method,Dataset,vs_Primary,vs_Independent,Difference"]

    # Combine all methods
    all_methods = {}
    for k, v in STARDIST_MODELS.items():
        all_methods[k] = v["display_name"]
    for k, v in DAPIEQ_METHODS.items():
        all_methods[k] = v["display_name"]

    for dataset_key, dataset_results in results["per_dataset"].items():
        display_name = DATASETS[dataset_key]["display_name"]

        for model_key, model_name in all_methods.items():
            if model_key in dataset_results["vs_primary"]:
                acc_primary = dataset_results["vs_primary"][model_key]["accuracy"]
                acc_independent = dataset_results["vs_independent"][model_key]["accuracy"]
                diff = acc_primary - acc_independent

                lines.append(
                    f"{model_name},{display_name},{acc_primary:.3f},{acc_independent:.3f},{diff:+.3f}"
                )

    # Add inter-annotator agreement for reference
    interannotator = {
        "ht1080_20x": 0.805,
        "ht1080_40x": 0.898,
    }
    for dataset_key, agreement in interannotator.items():
        display_name = DATASETS[dataset_key]["display_name"]
        lines.append(f"Inter-annotator,{display_name},{agreement:.3f},{agreement:.3f},0.000")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
