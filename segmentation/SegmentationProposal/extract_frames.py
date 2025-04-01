import yaml
import numpy as np
from aicsimageio import AICSImage
import scipy.ndimage as ndimage
from skimage.io import imsave
import os
from basicpy import BaSiC

# choose an appropriate scaling factor
# 20x - 1.0, 40x - 2.0, 100x - 5.0

scaling_factor = 2.0
# selects all frames if None
# otherwise use a list
selected_frames = [30] 
selected_channels = ["cyan", "magenta", "tubulin"]

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

cwd = os.getcwd()
dir_name = cwd.split(os.sep)[-2]
sample_number = cwd.split(os.sep)[-1]
channels = metadata["channels"]
if not set(selected_channels).issubset(channels):
    raise RuntimeError("Selected channels are not in video or provided metadata is wrong")

training_dir = "for_training"
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(os.path.join(training_dir, "images")):
    os.mkdir(os.path.join(training_dir, "images"))
if not os.path.isdir(os.path.join(training_dir, "masks")):
    os.mkdir(os.path.join(training_dir, "masks"))

# perform flatfield correction
flatfield_correction = True
for channel in selected_channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")

channel_idxs = [int(channels[channel]) for channel in selected_channels]

image_file = metadata["filename"]
labels_file = "dapieq_labels_manual.tif"


image = AICSImage(image_file)
labels = AICSImage(labels_file)

print(channel_idxs, selected_frames)
print(image.dims)
print(labels.dims)
if selected_frames is None:
    selected_frames = list(range(image.dims.T))
len_time = len(selected_frames) 
if not image.dims.T == labels.dims.Z:
    raise ValueError("Labels and image do not have same time scale")
image_data = np.zeros(shape=(len_time, len(channel_idxs), image.dims.Y, image.dims.X))
for idx, channel_idx in enumerate(channel_idxs):
    image_data_tmp = image.get_image_dask_data("TYX", C=channel_idx, T=selected_frames)
    image_data_tmp = image_data_tmp.compute()
    if flatfield_correction:
        basic_model = BaSiC.load_model(f"basic_model_{selected_channels[idx]}")
        corrected = basic_model.transform(image_data_tmp, timelapse=False)
        image_data_tmp = corrected.squeeze()

    # copy to right place
    image_data[:,  idx, :] = image_data_tmp[:]
label_data = labels.get_image_dask_data("ZYX", Z=selected_frames)
label_data = label_data.compute()
# use only uint16 to save data
label_data = label_data.astype(np.uint16)

zoom = (1, 1.0 / scaling_factor, 1.0 / scaling_factor)
for idx, frame in enumerate(selected_frames):
    frame_image_data = image_data[idx]
    frame_label_data = label_data[idx]
    # downscale if needed
    if not np.isclose(scaling_factor, 1.0):
        frame_image_data = ndimage.zoom(frame_image_data, zoom=zoom, order=1)
        frame_label_data = ndimage.zoom(frame_label_data, zoom=(zoom[1], zoom[2]), order=0)
    imsave(os.path.join("for_training", "images", f"{dir_name}_{sample_number}_{frame}.tif"), frame_image_data) 
    imsave(os.path.join("for_training", "masks", f"{dir_name}_{sample_number}_{frame}.tif"), frame_label_data, compression="zlib") 
