# Finetune Cellpose-SAM for FUCCI data

## Data preparation

Take the FUCCI data and re-format it using
the `prepare_cellpose_SAM_data.py` script.
**Attention**: This script has not been tested, for the preparation
of the paper I used the dataset that was prepared using the script in
the `classification/cellpose_sam` folder. This script works, this one here
might work.

## Training

Run `train_Cellpose-SAM.py`.
If the script does not work, make the batch size smaller.

## Validation

Run `validate_cellpose_sam_all_channels.py`, which gives you
values that can be compared to the other segmentation networks.

We also checked the pure segmentation performance of the 
Cellpose-SAM instance that was trained to segment and classify
at the same time. You can repeat this experiment by running
`validate_cellpose_sam_classifier_all_channels.py`
