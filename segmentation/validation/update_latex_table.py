#!/usr/bin/env python
"""
Update the LaTeX segmentation table with accuracy values from validation scripts.

This script runs all validation scripts and extracts the accuracy at IoU=0.5,
then updates the LaTeX table automatically.

Usage:
    python update_latex_table.py [--dry-run]
    python update_latex_table.py --env-config environments.json

Options:
    --dry-run       Show what would be updated without actually running validations
    --env-config    Path to JSON file specifying Python interpreters for each framework

Environment Configuration:
    Different scripts require different virtual environments (StarDist uses TensorFlow,
    InstanSeg and Cellpose use PyTorch). Create an environments.json file:

    {
        "stardist": "/path/to/stardist_env/bin/python",
        "instanseg": "/path/to/instanseg_env/bin/python",
        "cellpose": "/path/to/cellpose_env/bin/python",
        "confluentfucci": "/path/to/confluentfucci_env/bin/python"
    }

    On Windows, use paths like:
    {
        "stardist": "C:/Users/username/micromamba/envs/stardist_env/python.exe",
        "instanseg": "C:/Users/username/micromamba/envs/instanseg_env/python.exe",
        ...
    }

    Scripts will be run with the appropriate interpreter based on their framework.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.measure import label as label_skimage
from stardist import fill_label_holes
from tqdm import tqdm

from snr_utils import compute_snr_for_dataset

# Default Python interpreter (current environment)
DEFAULT_PYTHON = sys.executable

# Environment configuration - loaded from --env-config or defaults to current Python
ENV_CONFIG = {
    "stardist": DEFAULT_PYTHON,
    "instanseg": DEFAULT_PYTHON,
    "cellpose": DEFAULT_PYTHON,
    "confluentfucci": DEFAULT_PYTHON,
}

# Base directories
SCRIPT_DIR = Path(__file__).parent
LATEX_TABLE_PATH = SCRIPT_DIR / "latex_tables" / "table_segmentation.tex"
DATA_DIR = SCRIPT_DIR / ".." / ".." / "data"

# Dataset to row mapping in the LaTeX table
# Format: dataset_key -> (row_pattern, row_index)
DATASET_ROWS = {
    "validation": "Validation",
    "ht1080_20x": "HT1080, 20x",
    "ht1080_40x": "HT1080, 40x",
    "han_et_al": "Han et al.",
    "confluentfucci": "ConfluentFUCCI",
    "cellmaptracer": "CellMAPTracer",
    "cotton_et_al": "Cotton et al.",
}

# Column indices (0-based, after the dataset name column)
# Columns: SNR, rom1, rom2, rom3, rom4, rom5, rom6, rom7, rom8, 1-CH, 2-CH, 3-CH
COLUMN_INDICES = {
    "snr": 0,                       # SNR
    "dapieq_raw_stardist": 1,       # \rom{1}
    "dapieq_post_stardist": 2,      # \rom{2}
    "dapieq_post_cellpose": 3,      # \rom{3}
    "dapieq_cellpose_denoise": 4,   # \rom{4}
    "confluentfucci_method": 5,     # \rom{5}
    "instanseg_1ch": 6,             # \rom{6}
    "instanseg_2ch": 7,             # \rom{7}
    "instanseg_3ch": 8,             # \rom{8}
    "stardist_1ch": 9,              # 1-CH
    "stardist_2ch": 10,             # 2-CH
    "stardist_3ch": 11,             # 3-CH
}

# Mapping of validation scripts to (dataset, algorithm) pairs
# Format: (script_path_relative_to_validation, dataset_key, algorithm_key, special_parsing)
VALIDATION_SCRIPTS = [
    # Validation dataset (training data validation split)
    ("ValidationData/validate_all_networks.py", "validation", ["stardist_1ch", "stardist_2ch", "stardist_3ch"], "multi"),
    ("DAPI_equivalent/validate_network_2_channels_on_validation_data_without_preprocessing.py", "validation", "dapieq_raw_stardist", "single"),
    ("DAPI_equivalent/validate_network_2_channels_on_validation_data.py", "validation", "dapieq_post_stardist", "single"),
    ("DAPI_equivalent/validate_network_2_channels_on_validation_data_cellpose.py", "validation", "dapieq_post_cellpose", "single"),
    ("DAPI_equivalent/validate_network_2_channels_on_validation_data_cellpose_cyto3_denoise.py", "validation", "dapieq_cellpose_denoise", "single"),
    ("ConfluentFUCCI/confluent_fucci_on_validation_data.py", "validation", "confluentfucci_method", "single"),
    ("InstanSeg/validate_network_1_channel_on_validation_data.py", "validation", "instanseg_1ch", "single"),
    ("InstanSeg/validate_network_2_channels_on_validation_data.py", "validation", "instanseg_2ch", "single"),
    ("InstanSeg/validate_network_3_channels_on_validation_data.py", "validation", "instanseg_3ch", "single"),

    # HT1080 20x
    ("HT1080_extra_dataset/test_all_networks_20x.py", "ht1080_20x", ["stardist_1ch", "stardist_2ch", "stardist_3ch"], "multi"),
    ("HT1080_extra_dataset/dapieq_raw_20x.py", "ht1080_20x", "dapieq_raw_stardist", "single"),
    ("HT1080_extra_dataset/dapieq_20x.py", "ht1080_20x", "dapieq_post_stardist", "single"),
    ("HT1080_extra_dataset/dapieq_cellpose_20x.py", "ht1080_20x", "dapieq_post_cellpose", "single"),
    ("HT1080_extra_dataset/test_2_channel_confluentfucci_20x.py", "ht1080_20x", "confluentfucci_method", "single"),
    ("HT1080_extra_dataset/instanseg_1channel_20x.py", "ht1080_20x", "instanseg_1ch", "single"),
    ("HT1080_extra_dataset/instanseg_2channel_20x.py", "ht1080_20x", "instanseg_2ch", "single"),
    ("HT1080_extra_dataset/instanseg_3channel_20x.py", "ht1080_20x", "instanseg_3ch", "single"),

    # HT1080 40x
    ("HT1080_extra_dataset/test_all_networks_40x.py", "ht1080_40x", ["stardist_1ch", "stardist_2ch", "stardist_3ch"], "multi"),
    ("HT1080_extra_dataset/dapieq_raw.py", "ht1080_40x", "dapieq_raw_stardist", "single"),
    ("HT1080_extra_dataset/dapieq.py", "ht1080_40x", "dapieq_post_stardist", "single"),
    ("HT1080_extra_dataset/dapieq_cellpose.py", "ht1080_40x", "dapieq_post_cellpose", "single"),
    ("HT1080_extra_dataset/test_2_channel_confluentfucci.py", "ht1080_40x", "confluentfucci_method", "single"),
    ("HT1080_extra_dataset/instanseg_1channel.py", "ht1080_40x", "instanseg_1ch", "single"),
    ("HT1080_extra_dataset/instanseg_2channel.py", "ht1080_40x", "instanseg_2ch", "single"),
    ("HT1080_extra_dataset/instanseg_3channel.py", "ht1080_40x", "instanseg_3ch", "single"),

    # Han et al. (eDetectHaCaTFUCCI)
    ("eDetectHaCaTFUCCI/validate_networks_all_channels.py", "han_et_al", ["stardist_1ch", "stardist_2ch", "stardist_3ch"], "multi"),
    ("eDetectHaCaTFUCCI/validation_dapi_eq.py", "han_et_al", "dapieq_post_stardist", "single"),
    ("eDetectHaCaTFUCCI/validation_dapi_eq_without_preprocessing.py", "han_et_al", "dapieq_raw_stardist", "single"),
    ("eDetectHaCaTFUCCI/validation_dapi_eq_cellpose.py", "han_et_al", "dapieq_post_cellpose", "single"),
    ("eDetectHaCaTFUCCI/validate_2_channel_confluentfucci.py", "han_et_al", "confluentfucci_method", "single"),
    ("eDetectHaCaTFUCCI/validate_networks_instanseg_all_channels.py", "han_et_al", ["instanseg_1ch", "instanseg_2ch", "instanseg_3ch"], "multi_instanseg"),

    # ConfluentFUCCI test dataset
    ("ConfluentFUCCI/test_all_networks.py", "confluentfucci", ["stardist_1ch", "stardist_2ch", "stardist_3ch"], "multi"),
    ("ConfluentFUCCI/test_dapieq_without_preprocessing_on_confluentfucci.py", "confluentfucci", "dapieq_raw_stardist", "single"),
    ("ConfluentFUCCI/test_dapieq_on_confluent_fucci.py", "confluentfucci", "dapieq_post_stardist", "single"),
    ("ConfluentFUCCI/test_dapieq_cellpose_on_confluent_fucci.py", "confluentfucci", "dapieq_post_cellpose", "single"),
    ("ConfluentFUCCI/validate_2_channel_confluentfucci.py", "confluentfucci", "confluentfucci_method", "single"),
    ("ConfluentFUCCI/test_instanseg_on_confluentfucci.py", "confluentfucci", "instanseg_2ch", "single"),

    # CellMAPTracer
    ("CellMAPtracer/validation/test_2ch_network.py", "cellmaptracer", "stardist_2ch", "single"),
    ("CellMAPtracer/validation/validation_dapi_eq.py", "cellmaptracer", "dapieq_post_stardist", "single"),
    ("CellMAPtracer/validation/validation_dapi_eq_without_preprocessing.py", "cellmaptracer", "dapieq_raw_stardist", "single"),
    ("CellMAPtracer/validation/validation_dapi_eq_cellpose.py", "cellmaptracer", "dapieq_post_cellpose", "single"),
    ("CellMAPtracer/validation/validate_2_channel_confluentfucci.py", "cellmaptracer", "confluentfucci_method", "single"),
    ("CellMAPtracer/validation/test_2ch_network_instanseg.py", "cellmaptracer", "instanseg_2ch", "single"),

    # Cotton et al.
    ("CottonEtAl2024/validation/test_2ch_network.py", "cotton_et_al", "stardist_2ch", "single"),
    ("CottonEtAl2024/validation/validation_dapi_eq.py", "cotton_et_al", "dapieq_post_stardist", "single"),
    ("CottonEtAl2024/validation/validation_dapi_eq_without_preprocessing.py", "cotton_et_al", "dapieq_raw_stardist", "single"),
    ("CottonEtAl2024/validation/validation_dapi_eq_cellpose.py", "cotton_et_al", "dapieq_post_cellpose", "single"),
    ("CottonEtAl2024/validation/validate_2_channel_confluentfucci.py", "cotton_et_al", "confluentfucci_method", "single"),
    ("CottonEtAl2024/validation/test_2ch_network_instanseg.py", "cotton_et_al", "instanseg_2ch", "single"),
]

# Mapping from algorithm key to required framework/environment
# This determines which Python interpreter to use for each script
ALGORITHM_FRAMEWORK = {
    # StarDist-based (TensorFlow)
    "stardist_1ch": "stardist",
    "stardist_2ch": "stardist",
    "stardist_3ch": "stardist",
    "dapieq_raw_stardist": "stardist",
    "dapieq_post_stardist": "stardist",
    # Cellpose-based (PyTorch)
    "dapieq_post_cellpose": "cellpose",
    "dapieq_cellpose_denoise": "cellpose",
    # ConfluentFUCCI method (specific Cellpose version)
    "confluentfucci_method": "confluentfucci",
    # InstanSeg (PyTorch)
    "instanseg_1ch": "instanseg",
    "instanseg_2ch": "instanseg",
    "instanseg_3ch": "instanseg",
}


def get_framework_for_script(algo_keys) -> str:
    """
    Determine which framework/environment a script needs based on its algorithm keys.

    Parameters
    ----------
    algo_keys : str or list
        Algorithm key(s) for the script.

    Returns
    -------
    str
        Framework name: 'stardist', 'cellpose', 'confluentfucci', or 'instanseg'
    """
    if isinstance(algo_keys, str):
        return ALGORITHM_FRAMEWORK.get(algo_keys, "stardist")
    elif isinstance(algo_keys, list):
        # For multi-output scripts, check the first algorithm
        if algo_keys and algo_keys[0] in ALGORITHM_FRAMEWORK:
            return ALGORITHM_FRAMEWORK[algo_keys[0]]
    return "stardist"  # Default to stardist


def load_env_config(config_path: Path) -> None:
    """
    Load environment configuration from a JSON file.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file.
    """
    global ENV_CONFIG
    with open(config_path, "r") as f:
        loaded_config = json.load(f)

    # Update ENV_CONFIG with loaded values
    for key in ENV_CONFIG.keys():
        if key in loaded_config:
            python_path = loaded_config[key]
            # Validate that the Python interpreter exists
            if Path(python_path).exists():
                ENV_CONFIG[key] = python_path
                print(f"  {key}: {python_path}")
            else:
                print(f"  Warning: Python interpreter not found for {key}: {python_path}")
                print(f"           Using default: {DEFAULT_PYTHON}")


# Dataset paths for SNR computation
# Maps dataset_key to (images_dir, masks_dir, channel_order, relabel)
SNR_DATASET_PATHS = {
    "validation": {
        "type": "split",
        "data_dir": DATA_DIR / "training_data_tiled_strict_classified",
        "split_key": "validation",
        "channel_order": "YXC",
        "relabel": False,
    },
    "ht1080_20x": {
        "type": "directory",
        "images_dir": DATA_DIR / "data_set_HT1080_20x" / "images",
        "masks_dir": DATA_DIR / "data_set_HT1080_20x" / "masks",
        "channel_order": "YXC",
        "relabel": True,
    },
    "ht1080_40x": {
        "type": "directory",
        "images_dir": DATA_DIR / "data_set_HT1080_40x" / "images",
        "masks_dir": DATA_DIR / "data_set_HT1080_40x" / "masks",
        "channel_order": "YXC",
        "relabel": True,
    },
    "han_et_al": {
        "type": "directory",
        "images_dir": DATA_DIR / "HaCaT_Han_et_al" / "images",
        "masks_dir": DATA_DIR / "HaCaT_Han_et_al" / "masks",
        "channel_order": "YXC",
        "relabel": True,
    },
    "confluentfucci": {
        "type": "directory",
        "images_dir": DATA_DIR / "test_confluentfucci" / "images",
        "masks_dir": DATA_DIR / "test_confluentfucci" / "masks",
        "channel_order": "YXC",
        "relabel": True,
    },
    "cellmaptracer": {
        "type": "directory",
        "images_dir": DATA_DIR / "test_cellmaptracer" / "images",
        "masks_dir": DATA_DIR / "test_cellmaptracer" / "masks",
        "channel_order": "CYX",
        "relabel": True,
    },
    "cotton_et_al": {
        "type": "directory",
        "images_dir": DATA_DIR / "test_cottonetal" / "images",
        "masks_dir": DATA_DIR / "test_cottonetal" / "masks",
        "channel_order": "CYX",
        "relabel": True,
    },
}


def compute_snr_for_all_datasets(dry_run: bool = False) -> dict:
    """
    Compute mean SNR for each dataset.

    Returns
    -------
    dict
        Mapping of dataset_key to mean SNR value.
    """
    snr_results = {}

    for dataset_key, config in SNR_DATASET_PATHS.items():
        print(f"Computing SNR for {dataset_key}...")

        if dry_run:
            print(f"  Would compute SNR for {dataset_key}")
            continue

        try:
            if config["type"] == "split":
                # Load from dataset with split file
                data_dir = config["data_dir"]
                split_file = data_dir / "dataset_split.json"

                if not split_file.exists():
                    print(f"  Split file not found: {split_file}")
                    continue

                with open(split_file) as f:
                    split_data = json.load(f)

                filenames = split_data.get(config["split_key"], [])
                images = [imread(str(data_dir / "images" / fname)) for fname in filenames]
                masks = [fill_label_holes(imread(str(data_dir / "masks" / fname))) for fname in filenames]

            else:
                # Load from directory
                images_dir = config["images_dir"]
                masks_dir = config["masks_dir"]

                if not images_dir.exists() or not masks_dir.exists():
                    print(f"  Directory not found: {images_dir} or {masks_dir}")
                    continue

                image_files = sorted(glob(str(images_dir / "*.tif")))
                mask_files = sorted(glob(str(masks_dir / "*.tif")))

                if not image_files:
                    print(f"  No image files found in {images_dir}")
                    continue

                images = [imread(f) for f in image_files]
                masks = [imread(f) for f in mask_files]

                # Handle channel order
                if config["channel_order"] == "CYX":
                    images = [np.moveaxis(img, 0, -1) for img in images]

                # Relabel if needed
                if config.get("relabel", False):
                    masks = [fill_label_holes(label_skimage(m)) for m in masks]

            # Compute SNR
            snr_data = compute_snr_for_dataset(images, masks)

            if snr_data:
                # Get all SNR values and compute mean
                all_snr = []
                for img_data in snr_data.values():
                    for label_data in img_data.values():
                        if "mean_snr" in label_data and label_data["mean_snr"] is not None:
                            all_snr.append(label_data["mean_snr"])

                if all_snr:
                    mean_snr = np.mean(all_snr)
                    snr_results[dataset_key] = mean_snr
                    print(f"  Mean SNR: {mean_snr:.2f}")
                else:
                    print(f"  No valid SNR values computed")

        except Exception as e:
            print(f"  Error computing SNR: {e}")

    return snr_results


def parse_accuracy_from_output(output: str, parsing_mode: str) -> dict:
    """
    Parse accuracy values from script output.

    Parameters
    ----------
    output : str
        The stdout from running a validation script.
    parsing_mode : str
        'single' for scripts that output one accuracy value,
        'multi' for scripts that output 1-CH, 2-CH, 3-CH values,
        'multi_instanseg' for InstanSeg scripts with 1/2/3 channel output.

    Returns
    -------
    dict
        Mapping of algorithm key to accuracy value.
    """
    results = {}

    if parsing_mode == "single":
        # Look for: Stats at 0.5 IoU: DatasetMatching(..., accuracy=0.XX, ...)
        # or: Stats at 0.5 IoU:  DatasetMatching(...accuracy=0.XX...)
        match = re.search(r"Stats at 0\.5 IoU.*?accuracy=([0-9.]+)", output)
        if match:
            results["accuracy"] = float(match.group(1))

    elif parsing_mode == "multi":
        # Look for multiple lines like:
        # Stats at 0.5 IoU for 1 CH: ... accuracy=0.XX ...
        # Stats at 0.5 IoU for 2 CH: ... accuracy=0.XX ...
        # Stats at 0.5 IoU for 3 CH: ... accuracy=0.XX ...
        for ch in [1, 2, 3]:
            pattern = rf"Stats at 0\.5 IoU for {ch} CH.*?accuracy=([0-9.]+)"
            match = re.search(pattern, output)
            if match:
                results[f"{ch}ch"] = float(match.group(1))

    elif parsing_mode == "multi_instanseg":
        # Similar to multi but for InstanSeg output format
        # May need adjustment based on actual output format
        for ch in [1, 2, 3]:
            pattern = rf"Stats at 0\.5 IoU.*?{ch}.*?accuracy=([0-9.]+)"
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results[f"{ch}ch"] = float(match.group(1))

    return results


def run_validation_script(script_path: Path, framework: str = "stardist") -> str:
    """
    Run a validation script with the appropriate Python interpreter and return its output.

    Parameters
    ----------
    script_path : Path
        Path to the validation script.
    framework : str
        Framework name to determine which Python interpreter to use.
        One of: 'stardist', 'cellpose', 'confluentfucci', 'instanseg'

    Returns
    -------
    str
        Combined stdout and stderr from the script.
    """
    python_executable = ENV_CONFIG.get(framework, DEFAULT_PYTHON)
    try:
        result = subprocess.run(
            [python_executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=script_path.parent,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  Timeout running {script_path}")
        return ""
    except Exception as e:
        print(f"  Error running {script_path}: {e}")
        return ""


def update_latex_table(results: dict, latex_path: Path, dry_run: bool = False) -> None:
    """
    Update the LaTeX table with new accuracy values.

    Parameters
    ----------
    results : dict
        Nested dict: results[dataset][algorithm] = accuracy
    latex_path : Path
        Path to the LaTeX table file.
    dry_run : bool
        If True, print changes without writing.
    """
    with open(latex_path, "r") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Check if this line is a data row
        for dataset_key, row_name in DATASET_ROWS.items():
            if line.strip().startswith(row_name):
                # Parse and update this row
                new_line = update_row(line, row_name, dataset_key, results)
                if new_line != line:
                    print(f"  Updating row: {row_name}")
                    if dry_run:
                        print(f"    Old: {line.strip()}")
                        print(f"    New: {new_line.strip()}")
                new_lines.append(new_line)
                break
        else:
            new_lines.append(line)

    if not dry_run:
        with open(latex_path, "w") as f:
            f.write("\n".join(new_lines))
        print(f"\nUpdated {latex_path}")


def update_row(line: str, row_name: str, dataset_key: str, results: dict) -> str:
    """Update a single row in the LaTeX table."""
    if dataset_key not in results:
        return line

    # Split the line by '&' to get columns
    parts = line.split("&")
    if len(parts) < 13:  # dataset name + 12 data columns
        return line

    dataset_results = results[dataset_key]

    # Map algorithm keys to column indices
    for algo_key, col_idx in COLUMN_INDICES.items():
        if algo_key in dataset_results:
            value = dataset_results[algo_key]
            if value is not None:
                # Format as 0.XX
                formatted = f" {value:.2f}"
                # Update the column (col_idx + 1 because first column is dataset name)
                parts[col_idx + 1] = formatted

    return "&".join(parts)


def collect_results(dry_run: bool = False, framework_filter: str = None) -> dict:
    """
    Run all validation scripts and collect results.

    Parameters
    ----------
    dry_run : bool
        If True, only show what would be run without executing.
    framework_filter : str, optional
        If provided, only run scripts for this framework.

    Returns
    -------
    dict
        Nested dict: results[dataset][algorithm] = accuracy
    """
    results = {}

    for script_rel_path, dataset_key, algo_keys, parsing_mode in VALIDATION_SCRIPTS:
        script_path = SCRIPT_DIR / script_rel_path

        if not script_path.exists():
            print(f"  Script not found: {script_rel_path}")
            continue

        # Determine which framework/environment to use
        framework = get_framework_for_script(algo_keys)
        python_exe = ENV_CONFIG.get(framework, DEFAULT_PYTHON)

        # Skip if filtering by framework and this doesn't match
        if framework_filter and framework != framework_filter:
            continue

        print(f"Processing: {script_rel_path}")
        print(f"  Framework: {framework} -> {python_exe}")

        if dry_run:
            print(f"  Would run: {script_path}")
            continue

        output = run_validation_script(script_path, framework=framework)
        parsed = parse_accuracy_from_output(output, parsing_mode)

        if not parsed:
            print(f"  No accuracy values found in output")
            continue

        # Initialize dataset results if needed
        if dataset_key not in results:
            results[dataset_key] = {}

        # Store results
        if parsing_mode == "single":
            if "accuracy" in parsed:
                if isinstance(algo_keys, str):
                    results[dataset_key][algo_keys] = parsed["accuracy"]
                    print(f"  {algo_keys}: {parsed['accuracy']:.2f}")
        else:
            # Multi-output script
            if isinstance(algo_keys, list):
                for i, algo_key in enumerate(algo_keys, start=1):
                    ch_key = f"{i}ch"
                    if ch_key in parsed:
                        results[dataset_key][algo_key] = parsed[ch_key]
                        print(f"  {algo_key}: {parsed[ch_key]:.2f}")

    return results


def save_results_json(results: dict, output_path: Path) -> None:
    """Save results to a JSON file for later use."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def load_results_json(input_path: Path) -> dict:
    """Load results from a JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Update LaTeX segmentation table")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without running validations")
    parser.add_argument("--from-json", type=str,
                        help="Load results from a JSON file instead of running scripts")
    parser.add_argument("--save-json", type=str,
                        help="Save results to a JSON file")
    parser.add_argument("--skip-snr", action="store_true",
                        help="Skip SNR computation")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy computation (only compute SNR)")
    parser.add_argument("--env-config", type=str,
                        help="Path to JSON file specifying Python interpreters for each framework")
    parser.add_argument("--framework", type=str, choices=["stardist", "instanseg", "cellpose", "confluentfucci"],
                        help="Only run scripts for a specific framework (useful for running in stages)")
    args = parser.parse_args()

    print("=" * 60)
    print("LaTeX Table Updater for Segmentation Results")
    print("=" * 60)

    # Load environment configuration if provided
    if args.env_config:
        print(f"\nLoading environment configuration from {args.env_config}")
        load_env_config(Path(args.env_config))
    else:
        print("\nUsing default Python interpreter for all frameworks:")
        print(f"  {DEFAULT_PYTHON}")
        print("\nTip: Use --env-config to specify different Python interpreters for each framework.")

    if args.from_json:
        print(f"\nLoading results from {args.from_json}")
        results = load_results_json(Path(args.from_json))
    else:
        results = {}

        # Compute SNR for all datasets
        if not args.skip_snr:
            print("\nComputing SNR for all datasets...")
            snr_results = compute_snr_for_all_datasets(dry_run=args.dry_run)

            # Merge SNR results into main results
            for dataset_key, snr_value in snr_results.items():
                if dataset_key not in results:
                    results[dataset_key] = {}
                results[dataset_key]["snr"] = snr_value

        # Collect accuracy results from validation scripts
        if not args.skip_accuracy:
            print("\nCollecting accuracy results from validation scripts...")
            if args.framework:
                print(f"Filtering to framework: {args.framework}")
            print("This may take a while as each script needs to run.\n")
            accuracy_results = collect_results(dry_run=args.dry_run, framework_filter=args.framework)

            # Merge accuracy results into main results
            for dataset_key, algo_results in accuracy_results.items():
                if dataset_key not in results:
                    results[dataset_key] = {}
                results[dataset_key].update(algo_results)

    if args.save_json and not args.dry_run:
        save_results_json(results, Path(args.save_json))

    if not args.dry_run or args.from_json:
        print("\nUpdating LaTeX table...")
        update_latex_table(results, LATEX_TABLE_PATH, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == "__main__":
    main()
