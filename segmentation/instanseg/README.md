# InstanSeg Segmentation

This folder contains scripts to train InstanSeg for nuclear segmentation
on FUCCI microscopy data.

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r ../../requirements_instanseg_cellpose_sam.txt
```
This installs InstanSeg.

### 2. Clone InstanSeg Repository

Clone the InstanSeg repository (we use v0.0.9 from PyPi, which is tagged as 0.1.0)
because our process relies on their repository data structure:

```bash
cd ~/Documents/github  # or your preferred location
git clone --branch v0.1.0 https://github.com/instanseg/instanseg.git
cd instanseg
```

## Training Data Structure

The training data should follow the standard DeepFUCCI format:

```
FUCCI_data/
├── images/
│   ├── image_001.tif    # 3-channel FUCCI images (HxWxC)
│   └── ...
├── masks/
│   ├── image_001.tif    # Segmentation masks (same filenames)
│   └── ...
└── dataset_split.json   # Train/validation/test split
```

The `dataset_split.json` should contain:

```json
{
    "training": ["image_001.tif", ...],
    "validation": ["image_010.tif", ...],
    "test": ["image_020.tif", ...]  // optional
}
```

## Step-by-Step Training

### 1. Set Up Data Directory

Copy or symlink your FUCCI data to the InstanSeg `Raw_Datasets` folder:

```bash
# Create the directory structure
mkdir -p ~/Documents/github/instanseg/Raw_Datasets

# Create symlink to your training data
ln -s /path/to/your/training_data ~/Documents/github/instanseg/Raw_Datasets/FUCCI_data
```

### 2. Prepare the Dataset

Copy the dataset loading script to the InstanSeg notebooks directory and run it:

```bash
cp load_custom_dataset.py ~/Documents/github/instanseg/notebooks/
cd ~/Documents/github/instanseg/notebooks
python load_custom_dataset.py
```

This creates three dataset files in `instanseg/datasets/`:
- `fucci_1_channels_dataset.pth` (tubulin channel only)
- `fucci_2_channels_dataset.pth` (cyan + magenta)
- `fucci_3_channels_dataset.pth` (all three channels)

### 3. Train the Models

Navigate to the InstanSeg scripts directory and train:

```bash
cd ~/Documents/github/instanseg/instanseg/scripts

# Train 1-channel network
python train.py -data fucci_1_channels_dataset.pth -dim_in 1 \
    -source "[FUCCI_1CH_Dataset]" --experiment_str fucci_1ch -pixel_size 0.335

# Train 2-channel network
python train.py -data fucci_2_channels_dataset.pth -dim_in 2 \
    -source "[FUCCI_2CH_Dataset]" --experiment_str fucci_2ch -pixel_size 0.335

# Train 3-channel network
python train.py -data fucci_3_channels_dataset.pth -dim_in 3 \
    -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_3ch -pixel_size 0.335
```

For channel-invariant training (can handle any number of channels):

```bash
python train.py -ci True -data fucci_3_channels_dataset.pth \
    -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_channel_invariant
```

### 4. Test and Export Models

Set up environment variables (adjust paths to match your setup):

```bash
export INSTANSEG_MODEL_PATH=~/Documents/github/instanseg/instanseg/models
export INSTANSEG_BIOIMAGEIO_PATH=~/Documents/github/instanseg/instanseg/bioimageio_models
export INSTANSEG_TORCHSCRIPT_PATH=~/Documents/github/instanseg/instanseg/torchscripts
export EXAMPLE_IMAGE_PATH=~/Documents/github/instanseg/instanseg/examples
```

Run hyperparameter optimization and testing:

```bash
cd ~/Documents/github/instanseg/instanseg/scripts

# Optimize hyperparameters on validation set
python test.py --model_folder fucci_3ch -set Validation \
    --optimize_hyperparameters True --dataset fucci_3_channels

# Test and export model
python test.py --model_folder fucci_3ch -set Test --params best_params \
    --dataset fucci_3_channels -export_to_bioimageio True -export_to_torchscript True
```

## Validate on DeepFUCCI Dataset

After training, validate performance using the scripts in this folder:

```bash
cd /path/to/DeepFUCCI/segmentation/instanseg

# Validate all channel configurations
python validate_all_networks.py

# Validate channel-invariant network
python validate_channel_invariant_network.py
```

## Troubleshooting

**Dataset not found**: Ensure `FUCCI_data` is correctly symlinked in `Raw_Datasets/`
and contains the expected folder structure.

**Training fails with dimension error**: Check that images are in HxWxC format
(height x width x channels), not CxHxW.

**Environment variables not set**: The test.py script requires the environment
variables listed above. Add them to your `.bashrc` or set them before running.
