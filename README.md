# DeepFUCCI: Tools to use deep learning for bioimage analysis of FUCCI data

## Quickstart

To use the pretrained models, make sure that the right libraries are installed
(StarDist, InstanSeg, Cellpose-SAM, e.g.) and download the weights from
[Zenodo](https://zenodo.org/records/16574478?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZjNWE2ZGVjLWM1YzYtNGE1OS1iMjU1LWI0ZTMyNjY4MzViZCIsImRhdGEiOnt9LCJyYW5kb20iOiI0NWFhZmIxN2JmZGZkYTM0OWY0MmYxYWIyM2M2N2Q1OCJ9.LKoGQ880eIhgRcvUWQu-RfIu9aExpbJ2J43xk09THstPnmMduna9nqfl-kAM0PFjYrcfkAYhXIazBU7mSitRCg)

### Automated Download

You can download pretrained models and training data using the provided script:

```bash
pip install requests tqdm

# List available files
python download_data.py --list

# Download pretrained models (extracts to ~/models/)
python download_data.py --models-only

# Download training data (extracts to ./training_data/)
python download_data.py --data-only

# Download everything
python download_data.py
```

### Using Pretrained Models

Take the files `segment_and_classify_nuclei.py` and `segment_nuclei.py` from the `DataPreparation` folder as examples
how to use the StarDist networks.
They require the image file to have the correct pixel size. If the segmentation result is not satisfying, please check whether
the image metadata is correct.

Find visualization examples using Napari in the `Utilities` folder.

## Repository Structure

```
DeepFUCCI/
├── DataPreparation/       # Scripts to prepare training data from raw images
├── segmentation/          # Segmentation model training and validation
│   ├── training/          # StarDist training scripts
│   ├── instanseg/         # InstanSeg training scripts
│   ├── cellpose_sam/      # Cellpose-SAM training scripts
│   └── validation/        # Validation on various datasets
├── classification/        # Cell cycle phase classification
│   ├── training_scripts/  # StarDist classifier training
│   ├── cellpose_sam/      # Cellpose-SAM classifier training
│   └── validation/        # Classification validation
├── Tracking/              # Post-tracking analysis with fucciphase
└── Utilities/             # Visualization helpers
```

## Data Flow Overview

The diagram below shows how data flows from raw images to trained models:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                   │
│                        (DataPreparation folder)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw FUCCI Video                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌────────────────────────────┐                                              │
│  │ Basic-Flatfield-Correction │  (optional, for 20x/40x)                     │
│  └────────────┬───────────────┘                                              │
│               ▼                                                              │
│  ┌────────────────────────────┐                                              │
│  │ segment_nuclei.py          │  Initial segmentation with pretrained model  │
│  └────────────┬───────────────┘                                              │
│               ▼                                                              │
│  ┌────────────────────────────┐                                              │
│  │ view_in_napari.py          │  Manual curation in Napari                   │
│  └────────────┬───────────────┘                                              │
│               ▼                                                              │
│  ┌────────────────────────────┐     ┌─────────────────────────┐              │
│  │ extract_frames.py          │────▶│ for_training/           │              │
│  └────────────────────────────┘     │  ├── images/            │              │
│                                     │  └── masks/             │              │
│                                     └───────────┬─────────────┘              │
│                                                 ▼                            │
│  ┌────────────────────────────┐     ┌─────────────────────────┐              │
│  │ relabel_and_classify_      │────▶│ for_training_relabeled_ │              │
│  │ test_data.py               │     │ classified/             │              │
│  └────────────────────────────┘     │  ├── images/            │              │
│                                     │  ├── masks/             │              │
│                                     │  └── classes/           │              │
│                                     └───────────┬─────────────┘              │
│                                                 ▼                            │
│                                     ┌─────────────────────────┐              │
│                                     │ (manual) copy to:       │              │
│                                     │ training_data_relabeled_│              │
│                                     │ classified/             │              │
│                                     └───────────┬─────────────┘              │
│                                                 ▼                            │
│  ┌────────────────────────────┐     ┌─────────────────────────┐              │
│  │ tile_training_data.py      │────▶│ training_data_tiled_    │              │
│  └────────────────────────────┘     │ strict_classified/      │              │
│                                     │  ├── images/  (256x256) │              │
│                                     │  ├── masks/             │              │
│                                     │  └── classes/           │              │
│                                     └───────────┬─────────────┘              │
│                                                 ▼                            │
│  ┌────────────────────────────┐     ┌─────────────────────────┐              │
│  │ split_training_set.py      │────▶│ dataset_split.json      │              │
│  └────────────────────────────┘     └─────────────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Training scripts expect data in: ../training_data/                          │
│  (relative to the training script location)                                  │
│                                                                              │
│  Required structure:                                                         │
│  training_data/                                                              │
│  ├── images/*.tif          # 3-channel FUCCI images                          │
│  ├── masks/*.tif           # Segmentation masks                              │
│  ├── classes/*.json        # Cell cycle phase labels (for classifier)        │
│  └── dataset_split.json    # Train/validation split                          │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │ StarDist        │  │ InstanSeg       │  │ Cellpose-SAM    │               │
│  │ segmentation/   │  │ segmentation/   │  │ segmentation/   │               │
│  │ training/       │  │ instanseg/      │  │ cellpose_sam/   │               │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘               │
│           │                    │                    │                        │
│           ▼                    ▼                    ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐             │
│  │                    Trained Models                           │             │
│  │                                                             │             │
│  │  $HOME/models/                                              │             │
│  │  ├── stardist_1_channel_latest/                             │             │
│  │  ├── stardist_2_channel_latest/                             │             │
│  │  ├── stardist_3_channel_latest/                             │             │
│  │  ├── stardist_multiclass_1_channel_latest/                  │             │
│  │  ├── stardist_multiclass_2_channel_latest/                  │             │
│  │  └── stardist_multiclass_3_channel_latest/                  │             │
│  └─────────────────────────────────────────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Data Format

The training data can be downloaded from Zenodo [TODO insert link].
The data structure is:

```
training_data/
├── images/          # Flat-field corrected 3-channel frames (*.tif)
├── masks/           # Segmentation masks (*.tif)
├── classes/         # JSON files with phase labels (*.json)
└── dataset_split.json
```

### Channel Order

Images have three channels (in order):
1. **Cyan** (channel 0): G1 phase indicator
2. **Magenta** (channel 1): S/G2/M phase indicator
3. **Tubulin** (channel 2): Cytoplasmic stain (does not stain nucleus)

### Cell Cycle Phase Labels

The sensor can distinguish three cell cycle phases:

| Phase | Label | Description |
|-------|-------|-------------|
| G1 | **1** | G1 phase |
| G1/S | **2** | G1 to S transition |
| S/G2/M | **3** | S, G2, or M phase |

Labels are stored in JSON files where keys are mask labels and values are phase labels:

```json
{"1": 3, "2": 1, "3": 2, "4": 1, ...}
```

## Data Preparation

For the training, we annotated FUCCI images.
We used the multiplexed FUCCI sensor as described in
the [CALIPERS preprint](https://www.biorxiv.org/content/10.1101/2024.12.19.629259).
The images are flatfield-corrected if applicable (mostly for 20x and 40x acquisitions).

### Manual Annotation

To add your own data or reuse our scripts, follow the instructions in the
`DataPreparation` folder.

## Segmentation

The `segmentation` folder contains training scripts and instructions for the validation
and test of the trained network.

| Network | Folder | Classification Support |
|---------|--------|----------------------|
| StarDist | `segmentation/training/` | Yes (separate model) |
| InstanSeg | `segmentation/instanseg/` | No |
| Cellpose-SAM | `segmentation/cellpose_sam/` | Yes (joint model) |

## Classification

You can classify manually or use [fucciphase](https://github.com/Synthetic-Physiology-Lab/fucciphase.git)
to obtain an initial classification based on intensities.
More details can be found in the `DataPreparation` folder.

Training scripts can be found in the `classification` folder.

## Tracking

The segmented cells can be tracked, which yields cell-specific FUCCI intensities.
These can be postprocessed using the [fucciphase](https://github.com/synthetic-Physiology-Lab/fucciphase) package.
Examples are shown in the `Tracking` folder.

## Tested Networks

We (re-)trained the following networks and provide scripts for each:

| Network | Requirements | Capabilities |
|---------|--------------|--------------|
| StarDist | `requirements_stardist.txt` | Segmentation + Classification |
| Cellpose-SAM | `requirements_instanseg_cellpose_sam.txt` | Segmentation + Classification |
| InstanSeg | `requirements_instanseg_cellpose_sam.txt` | Segmentation only |
| ConfluentFUCCI | `requirements_confluentfucci.txt` | Segmentation only |

**Please feel free to share training recipes for other networks!**

## Windows Installation

TensorFlow is required for StarDist and has dropped GPU support on Windows.
Use the following recipe to run StarDist with GPU support on Windows:

1. Install Git Bash
2. Install micromamba as described here: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#umamba-install
3. Create an environment
   ```
   micromamba create -n stardist_env
   micromamba activate stardist_env
   ```
5. Make sure that the environment is active and run
   ```
   micromamba install python=3.10
   micromamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   python -m pip install numpy==1.26.4 "tensorflow<2.11"
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   **The last command should print something like:**
   ```
   [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```
   Then proceed to install StarDist and the other requirements (see `requirements_stardist.txt`).

## Known Issues

StarDist does not yet support NumPy v2.
If an error like

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
```

occurs, downgrade NumPy by running:
```
pip install numpy==1.26.4
```

## Cite Us

TODO
