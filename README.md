# DeepFUCCI: Tools to use deep learning for bioimage analysis of FUCCI data

## Data preparation

For the training, we annotated FUCCI images.
We used the multiplexed FUCCI sensor as described in
the [CALIPERS preprint](https://www.biorxiv.org/content/10.1101/2024.12.19.629259).
The images have three channels: cyan (indicator for G1 phase), magenta (indicator for S/G2/M phase),
and a cytoplasmic stain that does not stain the nucleus (here tubulin).
The images are flatfield-corrected if applicable (mostly for 20x and 40x acquisitions).

The training data can be downloaded from Zenodo [TODO insert link].
The data structure is:

1. `images`: flat-field corrected three channel frames (ending `*.tif`)
2. `masks`: segmentation masks (ending `*.tif`)
3. `classes`: JSON files with phase labels, more details below (ending `*.json`)

### Manual annotation

To add your own data or reuse our scripts, you can follow the instructions in the folder
`DataPreparation`.

## Segmentation

The `segmentation` folder contains training scripts and instructions for the validation
and test of the trained network.

## Classification

The sensor can distinguish three cell cycle phases:

1. G1 phase (label **1**)
1. G1/S phase (label **2**)
1. S/G2/M phase (label **3**)

Every nuclear label receives a cell cycle label.
These labels are stored in a JSON file.

You can classify manually or you use [fucciphase](https://github.com/Synthetic-Physiology-Lab/fucciphase.git)
to obtain an initial classification based on intensities.
More details can be found in the `DataPreparation` folder.

Scripts can be found in the `Classification` folder.

## Tracking

The segmented cells can be tracked, which yields cell-specific FUCCI intensities.
These can be postprocessed using the [fucciphase](https://github.com/synthetic-Physiology-Lab/fucciphase) package.
Examples are shown in the `Tracking` folder.

## Tested networks

We (re-)trained the following networks with our network and provide scripts for it:
* StarDist (see `requirements_stardist.txt` for used versions)
* Cellpose-SAM (see `requirements_instanseg_cellpose_sam.txt`, can be installed together)
* InstanSeg (only segmentation)
* ConfluentFUCCI (only segmentation, `requirements_confluentfucci.txt`, more details on installation in `Segmentation` folder)

**Please feel free to share training recipes for other networks!**

## Windows installation

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
## Known issues

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

## Cite us

TODO
