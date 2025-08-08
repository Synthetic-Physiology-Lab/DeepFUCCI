# Validation on validation data set

We use the validation data set (i.e., the part that was
randomly taken from the full training data set).
Here, the following validations were performed:

* `validate_all_networks.py`: The segmentation networks with 1, 2, and 3 input channels
  were validated. Shown in Fig. 1.
* `validate_all_classification_networks.py`: The segmentation performance of the classification networks
  with 1, 2, and 3 input channels was validated. Not shown.
* `validate_network_2_channels_flipped.py`: 
  Check the two channel network with swapped input channels.
