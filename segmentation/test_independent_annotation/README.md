# Independent Annotation Analysis

This folder contains scripts for analyzing inter-annotator agreement and comparing independent human annotations against ground truth masks.

## Purpose

The analysis quantifies how well independent human annotators agree with the ground truth annotations, providing a baseline for interpreting model performance. If human annotators disagree on 15% of labels, expecting model accuracy significantly above 85% may be unrealistic.

## Data Structure

The scripts expect independent annotations in a `masks_independent/` subfolder:

```
data/
├── data_set_HT1080_20x/
│   ├── images/                  # Original images
│   ├── masks/                   # Ground truth masks
│   └── masks_independent/       # Independent annotations
└── data_set_HT1080_40x/
    ├── images/
    ├── masks/
    └── masks_independent/
```

## Analysis Workflow

### Step 1: Find Disagreeing Labels

```bash
cd segmentation/test_independent_annotation
python find_disagreeing_labels.py
```

This script:
1. Compares each label in `masks_independent/` against ground truth in `masks/`
2. Computes IoU with overlapping ground truth labels
3. Labels with best IoU < 0.5 are considered disagreements
4. Saves masks containing only disagreeing labels to `masks_disagreement/`

**Output:**
- `data/data_set_HT1080_20x/masks_disagreement/*.tif`
- `data/data_set_HT1080_40x/masks_disagreement/*.tif`

### Step 2: Analyze IoU Distribution of Disagreements

```bash
python analyze_disagreement_iou.py
```

This script categorizes disagreements by their IoU values:
1. Computes IoU for each label against ground truth
2. Categorizes disagreements into types (no overlap, partial, near-match)
3. Generates visualization plots

**Output:**
- `disagreement_iou_analysis.pdf/png` - IoU distribution plots
- `disagreement_iou_results.json` - Detailed statistics

### Step 3: Analyze SNR of Disagreements

```bash
python analyze_disagreement_snr.py
```

This script analyzes whether disagreements are driven by low SNR (hard to see nuclei) or by semantic ambiguity:
1. Computes SNR for each label in independent annotations
2. Separates labels into agreeing (IoU >= 0.5) and disagreeing (IoU < 0.5)
3. Performs statistical tests comparing SNR distributions
4. Generates visualization plots

**Output:**
- `disagreement_snr_analysis.pdf/png` - Histogram and box plot
- `disagreement_snr_results.json` - Detailed statistics

### Step 4: Analyze FUCCI Channel Intensity

```bash
python analyze_disagreement_intensity.py
```

This script analyzes whether low cyan (G1) or magenta (S/G2/M) intensity contributes to disagreements:
1. Computes mean intensity in each FUCCI channel for each label
2. Compares intensity between agreeing and disagreeing labels
3. Breaks down by disagreement type (no overlap, partial, near-match)

**Output:**
- `disagreement_intensity_analysis.pdf/png` - Intensity comparison plots
- `disagreement_intensity_results.json` - Detailed statistics

## Results Summary

### Inter-Annotator Agreement

| Dataset    | Total Labels | Agreeing (IoU >= 0.5) | Disagreeing (IoU < 0.5) |
|------------|-------------|----------------------|------------------------|
| HT1080_20x | 807         | 650 (80.5%)          | 157 (19.5%)            |
| HT1080_40x | 756         | 679 (89.8%)          | 77 (10.2%)             |
| **Overall**| **1563**    | **1329 (85.0%)**     | **234 (15.0%)**        |

### Comparison with Model Performance

| Dataset    | Inter-Annotator Agreement | StarDist 2CH | StarDist 3CH |
|------------|--------------------------|--------------|--------------|
| HT1080_20x | 80.5%                    | 91.3%        | 91.1%        |
| HT1080_40x | 89.8%                    | 92.0%        | 93.8%        |

**Key Finding:** The StarDist models achieve higher agreement with ground truth than independent human annotators achieve with each other. This validates that the models are performing at or above human-expert level.

### IoU Distribution of Disagreements

| Disagreement Type | Count | Percentage | Interpretation |
|-------------------|-------|------------|----------------|
| No overlap (IoU=0) | 104 | 44.4% | Extra labels or completely missed cells |
| Partial overlap (0<IoU<0.3) | 73 | 31.2% | Likely merged/split cells |
| Near match (0.3≤IoU<0.5) | 57 | 24.4% | Boundary disagreements |

| Metric | Disagreeing | Agreeing |
|--------|-------------|----------|
| Mean IoU | 0.143 | 0.763 |
| Median IoU | 0.037 | 0.779 |

**Key Finding:** Almost half (44%) of disagreements have zero overlap—one annotator marked a nucleus that the other didn't consider a cell at all. Only 24% are near-matches with boundary disagreements.

### SNR Analysis of Disagreements

| Metric      | Agreeing Labels | Disagreeing Labels |
|-------------|----------------|-------------------|
| N           | 1329           | 234               |
| Mean SNR    | 2.13           | 2.17              |
| Median SNR  | 2.04           | 1.92              |
| Std SNR     | 0.89           | 1.31              |

**Statistical Tests:**
- T-test: t=-0.638, p=0.52 (not significant)
- Mann-Whitney U: U=160334, p=0.45 (not significant)

**Key Finding:** SNR is NOT the driver of inter-annotator disagreement. Disagreeing labels have essentially the same mean SNR as agreeing labels. This means disagreements reflect semantic ambiguity (interpretation differences) rather than visibility issues.

### FUCCI Channel Intensity Analysis

**Magenta intensity with different normalization methods:**

| Normalization Method | Agreeing | Disagreeing | Diff % | p-value |
|---------------------|----------|-------------|--------|---------|
| Raw intensity | 243.1 | 222.5 | -8.5% | **0.002** |
| Percentile normalized (0-1) | 0.332 | 0.277 | -16.6% | 0.065 |
| Relative to image mean | 1.194 | 1.020 | -14.5% | **0.00003** |

**Intensity by Disagreement Type (raw):**

| Category | N | Magenta Mean | vs Agreeing |
|----------|---|--------------|-------------|
| Agreeing | 1329 | 243.1 | - |
| No overlap (IoU=0) | 104 | 221.2 | p=0.01 * |
| Partial (0<IoU<0.3) | 73 | 221.6 | p=0.06 |
| Near match (0.3≤IoU<0.5) | 57 | 225.7 | p=0.34 |

**Key Findings:**
1. **Raw magenta** is significantly lower in disagreeing labels (p=0.002)
2. **Relative-to-mean normalization** shows the strongest effect (p=0.00003, 14.5% lower), confirming the effect is real and not an artifact of exposure differences
3. **Percentile normalization** weakens the effect (p=0.065), suggesting some contribution from per-image variation
4. **Near-match cases** (boundary disagreements) show NO intensity difference—these are purely geometric disagreements

### Method Accuracy vs Inter-Annotator Agreement

| Method | HT1080 20x | HT1080 40x |
|--------|------------|------------|
| *Inter-annotator* | *0.805* | *0.898* |
| DAPIeq raw StarDist | 0.361 | 0.851 |
| DAPIeq post StarDist | 0.852 | 0.854 |
| DAPIeq Cellpose | 0.770 | 0.804 |
| DAPIeq Cellpose Denoise | 0.782 | 0.745 |
| ConfluentFUCCI | 0.346 | 0.718 |
| InstanSeg 1CH | 0.000 | 0.001 |
| InstanSeg 2CH | 0.041 | 0.121 |
| InstanSeg 3CH | 0.256 | 0.398 |
| StarDist 1CH | 0.687 | 0.643 |
| **StarDist 2CH** | **0.913** | **0.920** |
| **StarDist 3CH** | **0.911** | **0.938** |

## Interpretation

1. **The 15% disagreement rate represents intrinsic annotation ambiguity.** This is not a data quality issue that could be fixed with better imaging—annotators disagree on interpretation, not visibility.

2. **Most disagreements are fundamental, not borderline.** 44% of disagreements have zero overlap (IoU=0), meaning one annotator marked something as a cell that the other completely ignored. Only 24% are near-matches with minor boundary differences.

3. **Low magenta intensity contributes to "missed" cells.** No-overlap disagreements (IoU=0) have 9% lower magenta intensity. These may be cells in early G1 phase with weak FUCCI signal, making them harder to identify as cells.

4. **Boundary disagreements are NOT intensity-driven.** Near-match cases (0.3≤IoU<0.5) show no significant intensity difference—these are purely geometric disagreements about where to draw boundaries.

5. **Model "errors" need reinterpretation.** When a model disagrees with ground truth on a clearly visible nucleus, it may be making a legitimate alternative interpretation rather than failing.

6. **The models exceed human reproducibility.** Since models achieve ~91-94% agreement while humans agree only 80-90%, the models are performing better than human inter-annotator variability allows.

7. **SNR explains dataset differences but not individual disagreements.** The 20x dataset has lower inter-annotator agreement (80.5% vs 89.8%) and lower SNR (1.77 vs 3.12), but within each dataset, disagreements are not concentrated in low-SNR nuclei.

## Scripts

| Script | Description |
|--------|-------------|
| `find_disagreeing_labels.py` | Find and save labels that disagree between annotations |
| `analyze_disagreement_iou.py` | Analyze IoU distribution and categorize disagreement types |
| `analyze_disagreement_snr.py` | Analyze SNR distribution of agreeing vs disagreeing labels |
| `analyze_disagreement_intensity.py` | Analyze FUCCI channel intensity (cyan/magenta) with normalization |
| `compare_methods_to_interannotator.py` | Compare method accuracies against inter-annotator agreement |
| `snr_utils.py` | Utility functions for SNR computation |

## Configuration

Edit the scripts to change:
- `IOU_THRESHOLD`: Default 0.5. Labels with IoU below this are considered disagreements.
- `DATASETS`: Dictionary of dataset names and paths to process.
- `DATA_DIR`: Base directory for data files.

## Requirements

```bash
pip install numpy scipy scikit-image matplotlib tqdm
```
