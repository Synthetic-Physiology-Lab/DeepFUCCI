# Data Preparation

This folder contains scripts to prepare training data from raw FUCCI microscopy images.

## Overview

The data preparation pipeline converts raw FUCCI videos into training-ready datasets:

```
Raw Video → Flatfield Correction → Segmentation → Manual Curation → Tiling → Split
```

## Output Folder Structure

Each step produces output in a specific folder:

| Step | Script | Output Folder |
|------|--------|---------------|
| 1. Extract frames | `extract_frames.py` | `for_training/` |
| 2. Classify | `relabel_and_classify_test_data.py` | `for_training_relabeled_classified/` |
| 3. Aggregate | (manual copy) | `training_data_relabeled_classified/` |
| 4. Tile | `tile_training_data.py` | `training_data_tiled_strict_classified/` |
| 5. Split | `split_training_set.py` | `dataset_split.json` |

The final `training_data_tiled_strict_classified/` folder (or a renamed/symlinked version)
is what the training scripts expect as `training_data/`.

## Prerequisites

We manage the scripts through a `metadata.yml` file where you need to enter:
- The filename
- The channel numbers
- The expected nuclear diameter (in microns)

## Step-by-Step Instructions

### Step 1: Flat-Field Correction (Optional)

If the image has notable flatfield artifacts (common for 20x and 40x acquisitions),
correct it using:

```bash
python Basic-Flatfield-Correction.py
```

This script produces a BaSiC model for each channel.

**Important**: The later scripts will use these models, so make sure that
the models in the folder belong to the data that you process!

### Step 2: Initial Segmentation

You have two options for initial segmentation:

#### Option A: DAPI-equivalent approach

Use `Nuclei-segmentation-max-projection.py` to segment a max-projection of the
two FUCCI channels. The image is denoised by Gaussian blur and background is
subtracted using a top-hat filter.

**Requires**: Expected diameter of the nucleus in the metadata YAML file.

#### Option B: Pre-trained network (Recommended)

Use `segment_nuclei.py` to segment using the pretrained StarDist networks:

```bash
python segment_nuclei.py
```

This produces three files: `stardist_labels_{1,2,3}_channel.tif`

**Note**: You need to enter the model name and reference pixel size in the script.
The default pixel size (0.3357 μm/pixel) is correct for the pretrained network.

To also predict cell cycle phases, use:

```bash
python segment_and_classify_nuclei.py
```

This saves masks in `stardist_labels_{1,2,3}_channel_classifier.tif` and
classes in `classes_{1,2,3}_channel.json`.

### Step 3: Manual Curation in Napari

Open the labels in Napari:

```bash
python view_in_napari.py
```

Then:
1. Duplicate the label layer with the best segmentation
2. Manually curate this layer (fix errors, remove artifacts)
3. Export the curated layer to `dapieq_labels_manual.tif`

**Tips**:
- Annotate entire videos because FUCCI sensors go dark after division
- Switch between frames to distinguish nuclei from debris/artifacts
- For large images, use `view_tiled_frame_in_napari.py` to work on tiles

### Step 4: Extract Training Frames

Export single frames in training-ready format:

```bash
python extract_frames.py
```

**Attention**: Make sure images are correctly scaled! Instructions are in the script.

**Output**: `for_training/` folder with `images/` and `masks/` subfolders.

### Step 5: Classify Nuclei

Install [fucciphase](https://github.com/Synthetic-Physiology-Lab/fucciphase) and run:

```bash
python relabel_and_classify_test_data.py
```

**Output**: `for_training_relabeled_classified/` with `images/`, `masks/`, and `classes/`.

### Step 6: Check Classifications

Copy `check_classifications.py` to the output folder and run it:

```bash
cp check_classifications.py for_training_relabeled_classified/
cd for_training_relabeled_classified
python check_classifications.py
```

This opens Napari showing image, masks, and classification proposals.

**To correct labels**:
1. Find the label ID by selecting it in Napari's label layer
2. Open the corresponding JSON file in `classes/`
3. Correct the entry for that label
4. Press Enter in the command line to proceed to the next file

**Note**: Delete old points layers as new ones are created for each file.

### Step 7: Aggregate Data

Copy verified files to the aggregation folder:

```bash
cp -r for_training_relabeled_classified/* training_data_relabeled_classified/
```

This folder holds all annotated data from multiple sessions before tiling.

### Step 8: Tile Training Data

Tile images to 256x256 pixels and filter out tiles with too few nuclei:

```bash
python tile_training_data.py
```

**Output**: `training_data_tiled_strict_classified/` with tiled `images/`, `masks/`, and `classes/`.

Tiles with fewer than 4 nuclei (not touching borders) are discarded.

### Step 9: Split Dataset

Split the dataset into training and validation sets:

```bash
python split_training_set.py
```

**Output**: `dataset_split.json` with lists of filenames for each split.

## Final Training Data Structure

After all steps, your training data should look like:

```
training_data/                      # Symlink or rename from training_data_tiled_strict_classified
├── images/
│   ├── image_001_1.tif            # 256x256, 3-channel
│   ├── image_001_2.tif
│   └── ...
├── masks/
│   ├── image_001_1.tif            # 256x256, uint16 labels
│   ├── image_001_2.tif
│   └── ...
├── classes/
│   ├── image_001_1.json           # {"1": 2, "2": 1, "3": 3, ...}
│   ├── image_001_2.json
│   └── ...
└── dataset_split.json             # {"training": [...], "validation": [...]}
```

## Connecting to Training Scripts

Training scripts expect data in `../training_data/` relative to their location.

**Option 1**: Rename the tiled folder
```bash
mv training_data_tiled_strict_classified ../training_data
```

**Option 2**: Create a symlink
```bash
ln -s $(pwd)/training_data_tiled_strict_classified ../training_data
```

See the main README for the complete data flow diagram.
