# Taken from InstanSeg github repository and modified
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
    data_path = os.path.abspath("../Raw_Datasets/FUCCI_data_binned")

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    items_1ch = []
    items_2ch = []
    items_3ch = []

    images_path = os.path.join(data_path, "images")
    masks_path = os.path.join(data_path, "masks")
    for file in sorted(os.listdir(images_path)):
        item_1ch = {}
        item_2ch = {}
        item_3ch = {}

        mask_path = os.path.join(masks_path, Path(file).name)
        image = io.imread(os.path.join(images_path, file))
        masks = io.imread(mask_path)
        # commented because messed up labels?
        # masks, _ = fastremap.renumber(masks, in_place=True)
        # masks = fastremap.refit(masks)
        assert masks.squeeze().ndim == 2, (
            r"The mask should be a 2D array, found {}".format(masks.shape)
        )

        # The masks should be a numpy array (or pytorch tensor) with shape (H, W). The values should be integers starting from 0. Each integer represents a different object.
        item_1ch["nucleus_masks"] = masks
        item_2ch["nucleus_masks"] = masks
        item_3ch["nucleus_masks"] = masks
        # The image is a numpy array with shape (H, W, C), C is the number of channels
        item_1ch["image"] = np.expand_dims(image[..., 2], axis=2)
        item_2ch["image"] = image[..., :2]
        item_3ch["image"] = image

        # Important, this is the handle to call the dataset when training.
        item_1ch["parent_dataset"] = "FUCCI_1CH_binned_Dataset"
        item_2ch["parent_dataset"] = "FUCCI_2CH_binned_Dataset"
        item_3ch["parent_dataset"] = "FUCCI_3CH_binned_Dataset"
        item_1ch["licence"] = "CC BY 4.0"
        item_2ch["licence"] = "CC BY 4.0"
        item_3ch["licence"] = "CC BY 4.0"

        # Pixel size should be in microns per pixel (usually it is in the range 0.2 to 1).
        # If the segmentation task is not for cells, or the pixel size is not known, you can comment this line out.
        # However, we strongly recommend you make sure the labels are of reasonable size, and fairly uniform across the dataset.
        # A good label area is around 300 pixels. See load_Cellpose in data_download.py for an example of how to load a dataset without pixel size.
        item_1ch["pixel_size"] = 0.6
        item_2ch["pixel_size"] = 0.6
        item_3ch["pixel_size"] = 0.6
        item_1ch["image_modality"] = "Fluorescence"
        item_2ch["image_modality"] = "Fluorescence"
        item_3ch["image_modality"] = "Fluorescence"
        item_1ch["file_name"] = file  # optional
        item_2ch["file_name"] = file  # optional
        item_3ch["file_name"] = file  # optional
        items_1ch.append(item_1ch)
        items_2ch.append(item_2ch)
        items_3ch.append(item_3ch)

    assert len(items_1ch) > 0, "No items found in the dataset folder."

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(items_1ch))
    n_train = max(1, int(round(0.8 * len(ind))))
    n_val = max(1, int(round(0.9 * len(ind))))
    Segmentation_Dataset_1ch["Train"] += list(map(lambda x: items_1ch[x], ind[:n_train]))
    Segmentation_Dataset_2ch["Train"] += list(map(lambda x: items_2ch[x], ind[:n_train]))
    Segmentation_Dataset_3ch["Train"] += list(map(lambda x: items_3ch[x], ind[:n_train]))

    Segmentation_Dataset_1ch["Validation"] += list(map(lambda x: items_1ch[x], ind[n_train:n_val]))
    Segmentation_Dataset_2ch["Validation"] += list(map(lambda x: items_2ch[x], ind[n_train:n_val]))
    Segmentation_Dataset_3ch["Validation"] += list(map(lambda x: items_3ch[x], ind[n_train:n_val]))

    Segmentation_Dataset_1ch["Test"] += list(map(lambda x: items_1ch[x], ind[n_val:]))
    Segmentation_Dataset_2ch["Test"] += list(map(lambda x: items_2ch[x], ind[n_val:]))
    Segmentation_Dataset_3ch["Test"] += list(map(lambda x: items_3ch[x], ind[n_val:]))

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
torch.save(Segmentation_Dataset_1ch, os.path.join(path, "fucci_1_channels_binned_dataset.pth"))

torch.save(Segmentation_Dataset_2ch, os.path.join(path, "fucci_2_channels_binned_dataset.pth"))

torch.save(Segmentation_Dataset_3ch, os.path.join(path, "fucci_3_channels_binned_dataset.pth"))


# To train a model on your custom dataset, run the following command in your terminal:
# ```
# cd instanseg/scripts
# python train.py -data custom_dataset.pth -source "[FUCCI_Dataset]"
# ```
