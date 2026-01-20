# Independent Annotation Testing

This folder contains scripts for comparing independent annotations against ground truth.

## Data Structure

The scripts expect independent annotations in a `masks_independent/` subfolder:

```
data/
├── data_set_HT1080_20x/
│   ├── images/
│   ├── masks/                  # Ground truth masks
│   └── masks_independent/      # Independent annotations
└── data_set_HT1080_40x/
    ├── images/
    ├── masks/                  # Ground truth masks
    └── masks_independent/      # Independent annotations
```

## Scripts

### `find_disagreeing_labels.py`

Compares independent annotations against ground truth and saves disagreeing labels.

**Usage:**
```bash
cd segmentation/test_independent_annotation
python find_disagreeing_labels.py
```

**How it works:**
1. For each label in `masks_independent/`, computes IoU with overlapping ground truth labels
2. Labels with best IoU < 0.5 are considered disagreements
3. Outputs a mask image containing only disagreeing labels to `masks_disagreement/`

**Output:**
- `data/data_set_HT1080_20x/masks_disagreement/*.tif`
- `data/data_set_HT1080_40x/masks_disagreement/*.tif`

Each output mask contains only the labels that disagreed with ground truth,
relabeled starting from 1.

## Configuration

Edit `find_disagreeing_labels.py` to change:
- `IOU_THRESHOLD`: Default 0.5. Labels with IoU below this are disagreements.
- `DATASETS`: Dictionary of dataset names and paths to process.
