# Taken from InstanSeg github repository and modified
import json
import os
import torch
import numpy as np
from skimage import io
from pathlib import Path


os.environ["INSTANSEG_RAW_DATASETS"] = os.path.abspath("../Raw_Datasets/")

if not os.path.exists(os.environ["INSTANSEG_RAW_DATASETS"]):
    os.mkdir(os.environ["INSTANSEG_RAW_DATASETS"])

os.environ["INSTANSEG_DATASET_PATH"] = os.path.abspath("../instanseg/datasets/")

if not os.path.exists(os.environ["INSTANSEG_DATASET_PATH"]):
    os.mkdir(os.environ["INSTANSEG_DATASET_PATH"])

Segmentation_Dataset = {}
Segmentation_Dataset["Train"] = []
Segmentation_Dataset["Test"] = []
Segmentation_Dataset["Validation"] = []


# Add your own dataset [Optional]
#
# Assuming your dataset is in this format:
# ```
# instanseg
# └───Raw_Datasets
#     └─── Nucleus_Segmentation
#         └───My_Own_Dataset
#             └─── images/*.tiff
#             └─── masks/*.tiff
# ```


def load_custom_dataset(
    Segmentation_Dataset_1ch, Segmentation_Dataset_2ch, Segmentation_Dataset_3ch
):
    data_path = os.path.abspath("../Raw_Datasets/FUCCI_data")

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    # Load dataset split from JSON (same as StarDist training scripts)
    dataset_split_path = os.path.join(data_path, "dataset_split.json")
    if not os.path.exists(dataset_split_path):
        raise FileNotFoundError(
            f"dataset_split.json not found at {dataset_split_path}. "
            "Please create this file with 'training', 'validation', and optionally 'test' keys."
        )

    with open(dataset_split_path) as fp:
        dataset_split = json.load(fp)

    images_path = os.path.join(data_path, "images")
    masks_path = os.path.join(data_path, "masks")

    def load_item(file):
        """Load a single image/mask pair and create items for all channel configurations."""
        item_1ch = {}
        item_2ch = {}
        item_3ch = {}

        mask_path = os.path.join(masks_path, Path(file).name)
        image = io.imread(os.path.join(images_path, file))
        masks = io.imread(mask_path)
        assert masks.squeeze().ndim == 2, (
            r"The mask should be a 2D array, found {}".format(masks.shape)
        )

        # The masks should be a numpy array (or pytorch tensor) with shape (H, W).
        # The values should be integers starting from 0. Each integer represents a different object.
        item_1ch["nucleus_masks"] = masks
        item_2ch["nucleus_masks"] = masks
        item_3ch["nucleus_masks"] = masks
        # The image is a numpy array with shape (H, W, C), C is the number of channels
        item_1ch["image"] = np.expand_dims(image[..., 2], axis=2)
        item_2ch["image"] = image[..., :2]
        item_3ch["image"] = image

        # Important, this is the handle to call the dataset when training.
        item_1ch["parent_dataset"] = "FUCCI_1CH_Dataset"
        item_2ch["parent_dataset"] = "FUCCI_2CH_Dataset"
        item_3ch["parent_dataset"] = "FUCCI_3CH_Dataset"
        item_1ch["licence"] = "CC BY 4.0"
        item_2ch["licence"] = "CC BY 4.0"
        item_3ch["licence"] = "CC BY 4.0"

        # Pixel size should be in microns per pixel (usually it is in the range 0.2 to 1).
        item_1ch["pixel_size"] = 0.3
        item_2ch["pixel_size"] = 0.3
        item_3ch["pixel_size"] = 0.3
        item_1ch["image_modality"] = "Fluorescence"
        item_2ch["image_modality"] = "Fluorescence"
        item_3ch["image_modality"] = "Fluorescence"
        item_1ch["file_name"] = file
        item_2ch["file_name"] = file
        item_3ch["file_name"] = file

        return item_1ch, item_2ch, item_3ch

    # Load training data
    for file in dataset_split["training"]:
        item_1ch, item_2ch, item_3ch = load_item(file)
        Segmentation_Dataset_1ch["Train"].append(item_1ch)
        Segmentation_Dataset_2ch["Train"].append(item_2ch)
        Segmentation_Dataset_3ch["Train"].append(item_3ch)

    # Load validation data
    for file in dataset_split["validation"]:
        item_1ch, item_2ch, item_3ch = load_item(file)
        Segmentation_Dataset_1ch["Validation"].append(item_1ch)
        Segmentation_Dataset_2ch["Validation"].append(item_2ch)
        Segmentation_Dataset_3ch["Validation"].append(item_3ch)

    # Load test data if present
    if "test" in dataset_split:
        for file in dataset_split["test"]:
            item_1ch, item_2ch, item_3ch = load_item(file)
            Segmentation_Dataset_1ch["Test"].append(item_1ch)
            Segmentation_Dataset_2ch["Test"].append(item_2ch)
            Segmentation_Dataset_3ch["Test"].append(item_3ch)

    assert len(Segmentation_Dataset_1ch["Train"]) > 0, "No training items found."

    return Segmentation_Dataset_1ch, Segmentation_Dataset_2ch, Segmentation_Dataset_3ch


Segmentation_Dataset_1ch = {}
Segmentation_Dataset_1ch["Train"] = []
Segmentation_Dataset_1ch["Test"] = []
Segmentation_Dataset_1ch["Validation"] = []

Segmentation_Dataset_2ch = {}
Segmentation_Dataset_2ch["Train"] = []
Segmentation_Dataset_2ch["Test"] = []
Segmentation_Dataset_2ch["Validation"] = []

Segmentation_Dataset_3ch = {}
Segmentation_Dataset_3ch["Train"] = []
Segmentation_Dataset_3ch["Test"] = []
Segmentation_Dataset_3ch["Validation"] = []

Segmentation_Dataset_1ch, Segmentation_Dataset_2ch, Segmentation_Dataset_3ch = (
    load_custom_dataset(
        Segmentation_Dataset_1ch, Segmentation_Dataset_2ch, Segmentation_Dataset_3ch
    )
)

# Save the custom dataset.
path = os.environ["INSTANSEG_DATASET_PATH"]

# You can change the name to whatever you want, but make sure it ends with "_dataset.pth".
torch.save(Segmentation_Dataset_1ch, os.path.join(path, "fucci_1_channels_dataset.pth"))

torch.save(Segmentation_Dataset_2ch, os.path.join(path, "fucci_2_channels_dataset.pth"))

torch.save(Segmentation_Dataset_3ch, os.path.join(path, "fucci_3_channels_dataset.pth"))


# To train a model on your custom dataset, run the following command in your terminal:
# ```
# cd instanseg/scripts
# python train.py -data custom_dataset.pth -source "[FUCCI_Dataset]"
# ```
