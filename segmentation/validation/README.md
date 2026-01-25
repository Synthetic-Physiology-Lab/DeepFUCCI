# Benchmarking the custom DeepFUCCI networks

This folder contains scripts that were developed to obtain the segmentation accuracy
on various datasets.

The datsets are:

* `CellMAPtracer`: PIP-FUCCI dataset
* `CottonEtAl2024`: intestinal epithelial cells
* `eDetectHaCaTFUCCI`: high SNR HaCaT cells with SMAD reporter
* `TestData`: low SNR HT1080 data
* `ConfluentFUCCI`: confluent cells used for ConfluentFUCCI training
* `DAPI_equivalent`: conventional DAPI equivalent scripts, no data
* `InstanSeg`: only scripts to check InstanSeg performance
* `ValidationData`: scripts to measure performance on validation data set

## SNR Analysis Scripts

The following scripts analyze Signal-to-Noise Ratio (SNR):

* `snr_utils.py`: Utility module for SNR computation
* `snr_histogram.py`: Generate SNR distribution histograms across datasets
* `snr_accuracy_analysis.py`: Analyze detection accuracy vs SNR for StarDist models
* `snr_accuracy_dapieq_analysis.py`: Analyze accuracy vs SNR for DAPI-equivalent methods

**IMPORTANT - Channel Selection**: For FUCCI data with 3 channels, SNR analysis uses ONLY the nuclear channels:
- Channel 0: Cyan (G1 phase marker) - NUCLEAR
- Channel 1: Magenta (S/G2/M phase marker) - NUCLEAR
- Channel 2: Tubulin - CYTOPLASMIC (**EXCLUDED** from SNR analysis)

The tubulin channel is cytoplasmic and has no nuclear signal, so including it would distort the SNR analysis.
