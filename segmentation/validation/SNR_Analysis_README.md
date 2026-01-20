# SNR Analysis Scripts

This folder contains scripts for analyzing Signal-to-Noise Ratio (SNR) of nuclei
and its relationship to segmentation accuracy.

## SNR Definition

SNR is defined as:

```
SNR = (I_in - I_out) / std_in
```

where:
- `I_in`: mean intensity inside the nucleus
- `I_out`: mean intensity of the background (computed once per image)
- `std_in`: standard deviation of intensity inside the nucleus

The background intensity `I_out` is computed once per image using all pixels
where no nucleus is present (mask == 0). This single background value is then
used for all nuclei in that image.

Reference: https://imagej.net/plugins/trackmate/analyzers/#contrast--signalnoise-ratio

## Scripts

### `snr_utils.py`

Utility module containing SNR calculation functions:

- `compute_background_intensity()`: Compute mean background for an image (mask == 0)
- `compute_snr_for_label()`: Compute SNR for a single nucleus using precomputed background
- `compute_snr_for_image()`: Compute SNR for all nuclei in an image
- `compute_snr_for_dataset()`: Compute SNR statistics for a dataset

### `snr_histogram.py`

Generates histograms of SNR distribution across all datasets.

**Usage:**
```bash
cd segmentation/validation
python snr_histogram.py
```

**Outputs:**
- `snr_histogram_all_datasets.pdf/png`: Combined histogram and box plot
- `snr_histogram_per_channel.pdf/png`: Per-channel SNR histograms (if multichannel)
- `snr_statistics.json`: Summary statistics

### `snr_accuracy_analysis.py`

Analyzes how segmentation detection rate (recall) varies with SNR.

**Usage:**
```bash
cd segmentation/validation
python snr_accuracy_analysis.py
```

**Outputs:**
- `snr_accuracy_by_dataset.pdf/png`: Detection rate vs SNR per dataset (subplots)
- `snr_accuracy_comparison.pdf/png`: Comparison across datasets (3CH model)
- `snr_accuracy_results.json`: Detailed results per model and dataset

## Data Structure

All data is stored in `data/` at the repository root:

```
data/
├── training_data_tiled_strict_classified/   # Training/validation data
│   ├── images/
│   ├── masks/
│   └── dataset_split.json
├── test_confluentfucci/                     # ConfluentFUCCI test data
│   ├── images/
│   └── masks/
├── test_cellmaptracer/                      # CellMAPtracer test data
│   ├── images/
│   └── masks/
├── test_cottonetal/                         # Cotton et al. test data
│   ├── images/
│   └── masks/
├── data_set_HT1080_20x/                     # HT1080 20x magnification
│   ├── images/
│   └── masks/
├── data_set_HT1080_40x/                     # HT1080 40x magnification
│   ├── images/
│   └── masks/
└── HaCaT_Han_et_al/                         # HaCaT data from Han et al.
    ├── images/
    └── masks/
```

## Supported Datasets

| Dataset | Description |
|---------|-------------|
| Training | Training split from `training_data_tiled_strict_classified` |
| Validation | Validation split from `training_data_tiled_strict_classified` |
| ConfluentFUCCI | External test dataset |
| HT1080_20x | HT1080 cells at 20x magnification |
| HT1080_40x | HT1080 cells at 40x magnification |
| CellMAPtracer | CellMAPtracer dataset |
| CottonEtAl | Data from Cotton et al. |
| HaCaT_Han | HaCaT data from Han et al. |

## Adding New Datasets

Edit the `EXTERNAL_DATASETS` dictionary in the scripts:

```python
EXTERNAL_DATASETS = {
    "MyNewDataset": {
        "type": "directory",
        "images_dir": f"{DATA_DIR}/my_new_dataset/images",
        "masks_dir": f"{DATA_DIR}/my_new_dataset/masks",
        "relabel": True,  # True if masks are binary and need labeling
        "channel_order": "YXC",  # "YXC" (HxWxC) or "CYX" (CxHxW)
        "scale": 1.0,  # Optional: pixel size correction for predictions
    },
}
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `type` | `"directory"` for folder with images/*.tif |
| `images_dir` | Path to images folder |
| `masks_dir` | Path to masks folder |
| `relabel` | Set `True` if masks are binary and need `skimage.label()` |
| `channel_order` | `"YXC"` if channels are last axis, `"CYX"` if channels are first |
| `scale` | Pixel size correction factor for StarDist predictions |

## Requirements

Same dependencies as other validation scripts:
- numpy
- matplotlib
- scikit-image
- stardist
- csbdeep
- tqdm

## Model Configuration

The accuracy analysis script expects StarDist models at:
```
training_1_channel_stardist/models/stardist
training_2_channels_stardist/models/stardist
training_3_channels_stardist/models/stardist
```

Modify `MODEL_CONFIGS` in `snr_accuracy_analysis.py` to change model paths.
