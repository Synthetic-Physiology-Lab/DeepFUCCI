import yaml
from aicsimageio import AICSImage
from skimage.io import imsave, imread
import numpy as np
from skimage.measure import label, regionprops_table
from skimage.segmentation import relabel_sequential
import os
import pandas as pd
from fucciphase.phase import estimate_cell_phase_from_max_intensity
from fucciphase.sensor import get_pipfucci_default_sensor
from skimage.util import invert
import json

def convert_categories(category: str) -> int:
    if category == "G1":
        return 1
    elif category == "S":
        return 2
    elif category == "G2/M":
        return 3

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

channels = metadata["channels"]
filename = metadata["filename"]
img_stream = AICSImage(filename)
mask_file = "gt_last_frame.tif" 

image = img_stream.get_image_dask_data("YXS", S=[int(channels["cyan"]),int(channels["magenta"])] , T=-1)
image = image.compute()

nucleus_labels = imread(mask_file)
print(image.shape, nucleus_labels.shape)
new_labels = label(nucleus_labels)
new_labels = new_labels.astype(np.uint16)
new_labels, _, _ = relabel_sequential(new_labels)

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
print(data)
print(background)
sensor = get_pipfucci_default_sensor()
channel_names = ["mean_intensity-0", "mean_intensity-1"]
thresholds = [0.1, 0.1]


estimate_cell_phase_from_max_intensity(
    data,
    channel_names,
    sensor,
    background=background[:2],  # only first two channels are bg
    thresholds=thresholds,
)

data["phase"] = data["DISCRETE_PHASE_MAX"].map(convert_categories)
cls_dict = {}
for _, row in data[["label", "phase"]].iterrows():
    cls_dict[int(row["label"])] = int(row["phase"])
with open(os.path.join("gt_last_frame_relabeled.tif"), "w") as fp:
    json.dump(cls_dict, fp)
imsave("gt_last_frame_relabeled.tif", arr=new_labels, compression="zlib")
