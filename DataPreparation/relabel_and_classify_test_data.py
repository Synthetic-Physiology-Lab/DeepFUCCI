from skimage.io import imsave, imread
from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential
import shutil
import numpy as np
from glob import glob
from csbdeep.utils import Path
import os
import pandas as pd
from fucciphase.phase import estimate_cell_phase_from_max_intensity
from fucciphase.sensor import get_fuccisa_default_sensor
from skimage.util import invert
import json


def convert_categories(category: str) -> int:
    if category == "G1":
        return 1
    elif category == "G1/S":
        return 2
    elif category == "S/G2/M":
        return 3
 
data_dir = "for_training"
new_data_dir = "for_training_relabeled_classified"

minimum_required_masks = 3

if not os.path.isdir(new_data_dir):
    os.mkdir(new_data_dir)
if not os.path.isdir(os.path.join(new_data_dir, "images")):
    os.mkdir(os.path.join(new_data_dir, "images"))
if not os.path.isdir(os.path.join(new_data_dir, "masks")):
    os.mkdir(os.path.join(new_data_dir, "masks"))

image_files = sorted(glob(os.path.join(data_dir, "images", "*.tif")))
mask_files = sorted(glob(os.path.join(data_dir, "masks", "*.tif")))
assert all(Path(x).name == Path(y).name for x, y in zip(image_files, mask_files))
class_path = os.path.join(new_data_dir, "classes")
if os.path.isdir(class_path):
    shutil.rmtree(class_path)
os.mkdir(class_path)


number_of_masks = 0

# cyan and magenta info
channels = ["mean_intensity-0", "mean_intensity-1"]
sensor = get_fuccisa_default_sensor()
thresholds = [0.1, 0.1]

for image_file, mask_file in zip(image_files, mask_files):
    nucleus_labels = imread(mask_file)
    image = imread(image_file)
    new_labels = label(nucleus_labels)
    new_labels = new_labels.astype(np.uint16)
    new_labels, _, _ = relabel_sequential(new_labels)

    n_masks = len(np.unique(new_labels)) - 1
    if n_masks <= minimum_required_masks:
        print(f"Skipped {image_file}, less than {minimum_required_masks} masks")
        continue
    number_of_masks += n_masks 
    imsave(os.path.join(new_data_dir, "masks", Path(mask_file).name), new_labels, dtype=np.uint16, compression="zlib", check_contrast=False)
    imsave(os.path.join(new_data_dir, "images", Path(image_file).name), image, compression="zlib")

    props = regionprops_table(new_labels, image,
                                      properties=['label', 'mean_intensity'])
    data = pd.DataFrame(props)
    background_label = invert(new_labels > 0)
    background_label = background_label.astype(np.uint8)
    bg_props = regionprops_table(background_label, image,
                                         properties=['mean_intensity'])
    bg_props_df = pd.DataFrame(bg_props)
    # has only one entry per channel because background is ignored
    background = bg_props_df.iloc[0].to_numpy()
    estimate_cell_phase_from_max_intensity(
        data,
        channels,
        sensor,
        background=background[:2],  # only first two channels are bg
        thresholds=thresholds,
    )

    data["phase"] = data["DISCRETE_PHASE_MAX"].map(convert_categories)
    cls_dict = {}
    for _, row in data[["label", "phase"]].iterrows():
        cls_dict[int(row["label"])] = int(row["phase"])
    basename = os.path.basename(image_file)
    # write a json
    basename = basename.replace(".tif", ".json")
    with open(os.path.join(class_path, basename), "w") as fp:
        json.dump(cls_dict, fp)

print(f"The training dataset contains {number_of_masks} labels")
