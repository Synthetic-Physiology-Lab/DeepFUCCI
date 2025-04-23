# DeepFUCCI

Tools to use deep learning for FUCCI segmentation

# Data preparation

For the training, we need annotated FUCCI images.
We used the multiplexed FUCCI sensor as described in
the [CALIPERS preprint](https://www.biorxiv.org/content/10.1101/2024.12.19.629259).
The images have three channels: cyan (indicator for G1 phase), magenta (indicator for S/G2/M phase),
and a cytoplasmic stain that does not stain the nucleus (here tubulin).
The images are flatfield-corrected if applicable (mostly for 20x and 40x acquisitions).

The data structure is:

1. `images`: flat-field corrected three channel frames (ending `*.tif`)
2. `masks`: segmentation masks (ending `*.tif`)
3. `classes`: JSON files with phase labels, more details below (ending `*.json`)

## Segmentation

A summary, related scripts, and validation data can be found in the
`segmentation` folder.

## Classification

The sensor can distinguish three cell cycle phases:

1. G1 phase (label **1**)
1. G1/S phase (label **2**)
1. S/G2/M phase (label **3**)

Every nuclear label receives a cell cycle label.
These labels are stored in a JSON file.

You can classify manually or you use [fucciphase](https://github.com/Synthetic-Physiology-Lab/fucciphase.git)
to obtain an initial classification based on intensities.
To manually check the annotations, you can run the script `check_classifications.py` (in folder `classification`).
It opens napari, loads the two FUCCI channels, the labels and the corresponding phase labels.
Once an image is inspected, hit enter in the terminal and the next image is opened.
Attention: The labels and image is automatically updated but the phase labels of the previous image
are not overwritten. Remove or hide them in napari.
If you find errors, edit the JSON file directly. 
The label number can be identified in napari.
Once all files are edited, you can close napari.

# Training

Having prepared the training data, a deep learning network can be trained.
In the following, the steps for individual architectures that we tested are documented.
**Please feel free to share training recipes for other networks!**

## StarDist

## InstanSeg

# Tracking

TODO
