from aicsimageio import AICSImage
from skimage.io import imread
import yaml
import napari
import os
from basicpy import BaSiC

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

filename = metadata["filename"]

viewer = napari.Viewer()

flatfield_correction = True
for channel in channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")


img_stream = AICSImage(filename)

# use get_image_dask_data if you have little RAM
img_tub = img_stream.get_image_data("TYX", C=int(channels["tubulin"]))
img_cyan = img_stream.get_image_data("TYX", C=int(channels["cyan"]))
img_magenta = img_stream.get_image_data("TYX", C=int(channels["magenta"]))

if flatfield_correction:
    basic_tubulin = BaSiC.load_model("basic_model_tubulin")
    basic_cyan = BaSiC.load_model("basic_model_cyan")
    basic_magenta = BaSiC.load_model("basic_model_magenta")
    img_tub = basic_tubulin.transform(img_tub, timelapse=True)
    img_cyan = basic_cyan.transform(img_cyan, timelapse=True)
    img_magenta = basic_magenta.transform(img_magenta, timelapse=True)
tub_layer=viewer.add_image(img_tub, colormap="green", blending="additive", name="tub")
viewer.add_image(img_cyan, colormap="cyan", blending="additive", name="cyan")
viewer.add_image(img_magenta, colormap="magenta", blending="additive", name="magenta")

labels_1 = imread("stardist_labels_1_channel.tif")
labels_2 = imread("stardist_labels_2_channel.tif")
labels_3 = imread("stardist_labels_3_channel.tif")

viewer.add_labels(labels_1, scale=tub_layer.scale)
viewer.add_labels(labels_2, scale=tub_layer.scale)
viewer.add_labels(labels_3, scale=tub_layer.scale)

napari.run()
