"""
Prepare FUCCI data for Cellpose-SAM segmentation training.

This script converts the standard DeepFUCCI training data format to the format
expected by Cellpose for training. The input data should have the following structure:

    training_data/
    ├── images/
    │   ├── image_001.tif
    │   └── ...
    ├── masks/
    │   ├── image_001.tif
    │   └── ...
    └── dataset_split.json

The output will be structured as:

    training_data_for_cellpose/
    ├── image_001.tif
    ├── image_001_masks.tif
    └── ...

Usage:
    python prepare_cellpose_SAM_data.py

After running this script, you can train with:
    python train_Cellpose-SAM.py
"""

from glob import glob
import os
from cellpose import io
import json
import tifffile
import fastremap
from pathlib import Path

logger, log_file = io.logger_setup()


def convert_fucci_data(data_dir: Path, out_dir: Path):
    """
    Convert DeepFUCCI format data to Cellpose training format.

    Parameters
    ----------
    data_dir : Path
        Path to the training data directory containing images/, masks/,
        and dataset_split.json
    out_dir : Path
        Output directory for Cellpose-formatted data
    """
    img_files = sorted(glob(os.path.join(data_dir, "images", "*.tif")))
    lbl_files = sorted(glob(os.path.join(data_dir, "masks", "*.tif")))

    if len(img_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {data_dir / 'images'}")

    if len(img_files) != len(lbl_files):
        raise ValueError(
            f"Number of images ({len(img_files)}) does not match "
            f"number of masks ({len(lbl_files)})"
        )

    dataset_split_path = data_dir / "dataset_split.json"
    if not dataset_split_path.exists():
        raise FileNotFoundError(
            f"dataset_split.json not found at {dataset_split_path}. "
            "This file should contain 'training' and 'validation' keys "
            "with lists of filenames."
        )

    with open(dataset_split_path) as fp:
        dataset_split = json.load(fp)

    out_dir.mkdir(exist_ok=True, parents=True)

    n_train = 0
    n_val = 0
    n_skipped = 0

    for img_file, lbl_file in zip(img_files, lbl_files):
        img_name = Path(img_file).name
        lbl_name = Path(lbl_file).name

        # Verify matching filenames
        if img_name != lbl_name:
            print(f"Warning: Image {img_name} does not match mask {lbl_name}, skipping")
            n_skipped += 1
            continue

        # Determine split
        if img_name in dataset_split.get("training", []):
            split = "train"
            n_train += 1
        elif img_name in dataset_split.get("validation", []):
            split = "val"
            n_val += 1
        else:
            print(f"Warning: {img_name} not in dataset_split.json, skipping")
            n_skipped += 1
            continue

        # Read data
        img = io.imread(img_file)
        masks = io.imread(lbl_file)

        # Ensure masks are properly labeled (sequential integers starting from 0)
        masks_relabeled = fastremap.renumber(masks.astype("uint16"))[0]

        # Output filename includes split prefix for organization
        out_name = f"{split}_{Path(img_file).stem}"

        # Save image and mask in Cellpose format
        tifffile.imwrite(
            out_dir / f"{out_name}.tif",
            data=img,
            compression="zlib",
        )
        tifffile.imwrite(
            out_dir / f"{out_name}_masks.tif",
            data=masks_relabeled,
            compression="zlib",
        )

    print(f"Conversion complete:")
    print(f"  Training images: {n_train}")
    print(f"  Validation images: {n_val}")
    print(f"  Skipped: {n_skipped}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    # Input: standard DeepFUCCI training data format
    # This should be a symlink or copy of the training_data folder
    fucci_data_dir = Path("training_data")

    # Output: Cellpose-compatible format
    # This matches the expected input for train_Cellpose-SAM.py
    out_dir = Path("training_data_for_cellpose")

    if not fucci_data_dir.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {fucci_data_dir}\n"
            "Please create a symlink or copy your training data:\n"
            f"  ln -s /path/to/your/training_data {fucci_data_dir}"
        )

    print(f"Converting data from {fucci_data_dir} to {out_dir}")
    convert_fucci_data(fucci_data_dir, out_dir)
