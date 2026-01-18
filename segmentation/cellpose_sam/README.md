# Finetune Cellpose-SAM for FUCCI Segmentation

This folder contains scripts to train Cellpose-SAM for nuclear segmentation
on FUCCI microscopy data.

## Prerequisites

Install the required packages:

```bash
pip install -r ../../requirements_instanseg_cellpose_sam.txt
```

Ensure you have GPU support configured for PyTorch.

## Training Data Structure

The training data should follow the standard DeepFUCCI format:

```
training_data/
├── images/
│   ├── image_001.tif    # 3-channel FUCCI images (HxWxC)
│   ├── image_002.tif
│   └── ...
├── masks/
│   ├── image_001.tif    # Segmentation masks (same filenames as images)
│   ├── image_002.tif
│   └── ...
└── dataset_split.json   # Train/validation split
```

The `dataset_split.json` file should contain:

```json
{
    "training": ["image_001.tif", "image_002.tif", ...],
    "validation": ["image_010.tif", "image_011.tif", ...]
}
```

## Step-by-Step Training

### 1. Prepare Training Data

First, create a symlink to your training data (or copy it):

```bash
cd segmentation/cellpose_sam
ln -s /path/to/your/training_data training_data
```

Then convert the data to Cellpose format:

```bash
python prepare_cellpose_SAM_data.py
```

This creates `training_data_for_cellpose/` with properly formatted files.

### 2. Train the Model

```bash
python train_Cellpose-SAM.py
```

Training parameters (can be adjusted in the script):
- `n_epochs`: 200 (default)
- `batch_size`: 4 (reduce if running out of GPU memory)
- `learning_rate`: 1e-5

The trained model will be saved to `~/.cellpose/models/FUCCI_cpsam`.

### 3. Validate the Model

After training, validate the model on the validation dataset:

```bash
python validate_cellpose_sam_all_channels.py
```

## Troubleshooting

**Out of GPU memory**: Reduce `batch_size` in `train_Cellpose-SAM.py` (e.g., from 4 to 2).

**No GPU detected**: Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Additional Scripts

- `validate_cellpose_sam_all_channels.py`: Validate segmentation-only model
- `validate_cellpose_sam_classifier_all_channels.py`: Validate segmentation
  performance of the classification model (trained in `classification/cellpose_sam/`)
