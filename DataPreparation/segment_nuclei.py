import os
from aicsimageio import AICSImage
import yaml
import numpy as np
from skimage.io import imsave
from stardist.models import StarDist2D
from tqdm import tqdm
import csbdeep
from basicpy import BaSiC
from pathlib import Path

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

flatfield_correction = True
for channel in channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")


filename = metadata["filename"]

img_stream = AICSImage(filename)
labels_1d = np.zeros(shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16)
labels_2d = np.zeros(shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16)
labels_3d = np.zeros(shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16)

model_1d = StarDist2D(None, name='stardist_1_channel_latest', basedir=Path.home() / 'models')
model_2d = StarDist2D(None, name='stardist_2_channel_latest', basedir=Path.home() / 'models')
model_3d = StarDist2D(None, name='stardist_3_channel_latest', basedir=Path.home() / 'models')

reference_pixel_size = 0.3357
actual_pixel_size = img_stream.physical_pixel_sizes.X
scale = actual_pixel_size / reference_pixel_size

if flatfield_correction:
    basic_tubulin = BaSiC.load_model("basic_model_tubulin")
    basic_cyan = BaSiC.load_model("basic_model_cyan")
    basic_magenta = BaSiC.load_model("basic_model_magenta")
for T in tqdm(range(img_stream.dims.T)):
    img_tubulin = img_stream.get_image_data("YX", C=int(channels["tubulin"]), T=T)
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)
    if flatfield_correction:
        corrected = basic_tubulin.transform(img_tubulin, timelapse=False)
        img_tubulin = corrected.squeeze()
        corrected = basic_cyan.transform(img_cyan, timelapse=False)
        img_cyan = corrected.squeeze()
        corrected = basic_magenta.transform(img_magenta, timelapse=False)
        img_magenta = corrected.squeeze()

    # normalize image
    img_tubulin = csbdeep.utils.normalize(img_tubulin, pmin=1)
    img_cyan = csbdeep.utils.normalize(img_cyan, pmin=1)
    img_magenta = csbdeep.utils.normalize(img_magenta, pmin=1)

    # 1 channel model
    labels, details = model_1d.predict_instances(img_tubulin, scale=scale)
    labels_1d[T] = labels[:]

    # 2 channel model
    labels, details = model_2d.predict_instances(np.moveaxis(np.stack([img_cyan, img_magenta]), 0, -1), scale=scale)
    labels_2d[T] = labels[:]

    # 3 channel model
    labels, details = model_3d.predict_instances(np.moveaxis(np.stack([img_cyan, img_magenta, img_tubulin]), 0, -1), scale=scale)
    labels_3d[T] = labels[:]

imsave("stardist_labels_3_channel.tif", labels_3d)
imsave("stardist_labels_2_channel.tif", labels_2d)
imsave("stardist_labels_1_channel.tif", labels_1d)
