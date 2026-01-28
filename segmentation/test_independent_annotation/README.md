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

**Note:** SNR is computed using only the nuclear channels (cyan=0, magenta=1). The tubulin channel (2) is cytoplasmic and excluded from SNR computation as it has no nuclear signal.

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

### Step 5: Validate Models Against Independent Annotations

```bash
conda activate kerasEnv
python validate_against_independent.py
```

This script tests whether models learned **generalizable nuclear features** or **annotation-specific biases** by validating against independent annotations:
1. Loads StarDist models (1CH, 2CH, 3CH)
2. Runs predictions on HT1080 20x and 40x test datasets
3. Computes accuracy against **both** mask sets:
   - `masks/` (primary annotator - used for training)
   - `masks_independent/` (independent annotator)
4. Compares the two accuracy values to detect annotation bias

**Output:**
- `validation_independent_results.json` - Full results with accuracy, F1, precision, recall
- `validation_independent_comparison.csv` - Side-by-side comparison table

### Step 6: Generate Extended Comparison Tables

```bash
python compare_methods_to_interannotator.py
```

This script now includes extended comparison showing accuracy against both annotators:

**Output:**
- `interannotator_comparison.csv` - Original comparison table
- `interannotator_comparison.tex` - LaTeX formatted table
- `interannotator_comparison_extended.csv` - Extended table with independent validation

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

### Model Validation Against Independent Annotations

This analysis tests whether models learned generalizable features or annotation-specific biases.

| Dataset | Model | vs Primary | vs Independent | Difference | Interpretation |
|---------|-------|------------|----------------|------------|----------------|
| HT1080 20x | StarDist 1CH | 68.7% | 61.2% | +7.5% | Some annotation bias |
| HT1080 20x | StarDist 2CH | 91.3% | 79.3% | +12.0% | Annotation bias detected |
| HT1080 20x | StarDist 3CH | 91.0% | 77.9% | +13.1% | Annotation bias detected |
| HT1080 20x | DAPIeq raw | 36.1% | 32.1% | +4.0% | Mild bias (but low accuracy) |
| HT1080 20x | DAPIeq post | 85.2% | 78.5% | +6.7% | Some annotation bias |
| HT1080 40x | StarDist 1CH | 64.3% | 59.7% | +4.7% | Mild annotation bias |
| HT1080 40x | StarDist 2CH | 92.0% | 86.4% | +5.7% | Mild annotation bias |
| HT1080 40x | StarDist 3CH | 93.8% | 84.9% | +8.9% | Some annotation bias |
| HT1080 40x | DAPIeq raw | 85.1% | 83.9% | +1.2% | **Generalizes well** |
| HT1080 40x | DAPIeq post | 85.4% | 87.0% | -1.6% | **Generalizes well** |

**Summary Statistics:**

| Model | Avg vs Primary | Avg vs Independent | Avg Difference |
|-------|----------------|-------------------|----------------|
| StarDist 1CH | 66.5% | 60.4% | +6.1% |
| StarDist 2CH | 91.7% | 82.9% | +8.8% |
| StarDist 3CH | 92.4% | 81.4% | +11.0% |
| DAPIeq raw | 60.6% | 58.0% | +2.6% |
| DAPIeq post | 85.3% | 82.7% | +2.6% |

**Key Findings:**

1. **Custom-trained models show annotation-specific bias (5-13%).** StarDist models perform better against the primary annotator (used in training) than against the independent annotator.

2. **DAPI-equivalent methods show minimal bias (+2.6%).** The pretrained StarDist model with DAPI-equivalent preprocessing generalizes well across annotators, suggesting it learned more universal nuclear features.

3. **Independent validation ≈ inter-annotator agreement.** When validated against independent annotations, model accuracy (77-87%) is close to the inter-annotator agreement level (80-90%), suggesting the models are learning something real but custom-trained models also pick up annotation-specific patterns.

4. **3-channel models show more bias than 2-channel.** StarDist 3CH shows larger differences (+11% avg) than StarDist 2CH (+8.8% avg), possibly because the additional tubulin channel provides cues that correlate with the primary annotator's style.

5. **40x dataset shows less bias.** The higher-quality 40x images show smaller differences than 20x images, suggesting annotation bias is more pronounced when image quality is lower.

6. **DAPIeq post actually performs BETTER against the independent annotator on 40x data (-1.6% difference).** This suggests the pretrained model may capture more generalizable features than the training data itself.

### SNR Analysis: Model Detections vs Misses

This analysis tests whether model failures (missed nuclei) correlate with low SNR.

| Model | Detected SNR | Missed SNR | Difference | p-value |
|-------|--------------|------------|------------|---------|
| DAPIeq raw | 2.788 | 1.130 | **+1.658** | 3.13e-194 |
| DAPIeq post | 2.279 | 0.939 | **+1.341** | 8.96e-40 |
| StarDist 2CH | 2.243 | 0.690 | **+1.553** | 1.74e-32 |

**Per-Dataset Breakdown:**

| Model | Dataset | Detected (n) | Missed (n) | Recall | Detected SNR | Missed SNR |
|-------|---------|--------------|------------|--------|--------------|------------|
| DAPIeq raw | HT1080 20x | 272 | 474 | 36.5% | 2.00 | 1.09 |
| DAPIeq raw | HT1080 40x | 646 | 88 | 88.0% | 3.12 | 1.32 |
| DAPIeq post | HT1080 20x | 674 | 72 | 90.3% | 1.52 | 0.52 |
| DAPIeq post | HT1080 40x | 673 | 61 | 91.7% | 3.04 | 1.44 |
| StarDist 2CH | HT1080 20x | 696 | 50 | 93.3% | 1.50 | 0.34 |
| StarDist 2CH | HT1080 40x | 704 | 30 | 95.9% | 2.98 | 1.27 |

**Key Findings:**

1. **Low SNR strongly drives model misses.** All models show highly significant differences (p < 10⁻³²) between detected and missed nuclei SNR. Missed nuclei have ~1.1-1.7 lower SNR on average.

2. **DAPIeq raw fails dramatically on low-SNR data (20x).** Only 36.5% recall on 20x vs 88% on 40x. The postprocessing (top-hat + blur) rescues this to 90.3%.

3. **StarDist 2CH has the lowest missed SNR (0.69).** This means StarDist can detect dimmer nuclei than DAPI-eq methods, explaining its higher overall recall (93-96%).

4. **All models struggle with the same low-SNR nuclei.** The missed nuclei have similar SNR across methods (~0.7-1.1), suggesting a fundamental visibility limit.

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
| Mean SNR    | 1.88           | 0.77              |
| Median SNR  | 1.77           | 0.60              |
| Std SNR     | 0.87           | 0.74              |

**Statistical Tests:**
- T-test: t=18.31, p=5.55e-68 (highly significant)
- Mann-Whitney U: U=264307, p=1.71e-65 (highly significant)

**Key Finding:** SNR IS a significant driver of inter-annotator disagreement. Disagreeing labels have substantially lower mean SNR (0.77 vs 1.88) than agreeing labels. Low nuclear signal makes nuclei harder to identify consistently, leading to annotation disagreements.

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

1. **Low nuclear SNR drives annotation disagreements.** Disagreeing labels have significantly lower SNR (mean 0.77) than agreeing labels (mean 1.88). Low nuclear signal makes cells harder to identify, leading to inconsistent annotations.

2. **Most disagreements are fundamental, not borderline.** 44% of disagreements have zero overlap (IoU=0), meaning one annotator marked something as a cell that the other completely ignored. Only 24% are near-matches with minor boundary differences.

3. **Low magenta intensity contributes to "missed" cells.** No-overlap disagreements (IoU=0) have 9% lower magenta intensity. These may be cells in early G1 phase with weak FUCCI signal, making them harder to identify as cells.

4. **Boundary disagreements are NOT intensity-driven.** Near-match cases (0.3≤IoU<0.5) show no significant intensity difference—these are purely geometric disagreements about where to draw boundaries.

5. **Model "errors" need reinterpretation.** When a model disagrees with ground truth on a low-SNR nucleus, it may be facing the same visibility challenges as human annotators.

6. **The models exceed human reproducibility.** Since models achieve ~91-94% agreement while humans agree only 80-90%, the models are performing better than human inter-annotator variability allows.

7. **SNR explains both dataset differences AND individual disagreements.** The 20x dataset has lower inter-annotator agreement (80.5% vs 89.8%) and lower SNR, and within each dataset, disagreements ARE concentrated in low-SNR nuclei (mean SNR 0.77 vs 1.88).

8. **Models show some annotation-specific bias.** When validated against independent annotations, accuracy drops by 5-13% compared to primary annotations. This suggests models learn both generalizable features AND some annotation-specific patterns.

9. **Independent validation ≈ inter-annotator agreement.** Model accuracy against independent annotations (77-86%) is close to inter-annotator agreement (80-90%), suggesting models perform at human-level when evaluated fairly against unseen annotation styles.

10. **Image quality affects annotation bias.** Higher quality 40x images show less annotation bias (5-9% difference) than 20x images (7-13% difference), suggesting clearer images lead to more consistent annotations and more generalizable model learning.

## Scripts

| Script | Description |
|--------|-------------|
| `find_disagreeing_labels.py` | Find and save labels that disagree between annotations |
| `analyze_disagreement_iou.py` | Analyze IoU distribution and categorize disagreement types |
| `analyze_disagreement_snr.py` | Analyze SNR distribution of agreeing vs disagreeing labels |
| `analyze_disagreement_intensity.py` | Analyze FUCCI channel intensity (cyan/magenta) with normalization |
| `validate_against_independent.py` | Validate models against independent annotator masks |
| `analyze_model_snr.py` | **Analyze SNR of detected vs missed nuclei for each model** |
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
