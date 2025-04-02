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

tiles_per_row = 3
frame = 40 

viewer = napari.Viewer()

flatfield_correction = True
for channel in channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")


img_stream = AICSImage(filename)

img_tub = img_stream.get_image_dask_data("YX", C=int(channels["tubulin"]), T=frame)
img_cyan = img_stream.get_image_dask_data("YX", C=int(channels["cyan"]), T=frame)
img_magenta = img_stream.get_image_dask_data("YX", C=int(channels["magenta"]), T=frame)

if flatfield_correction:
    basic_tubulin = BaSiC.load_model("basic_model_tubulin")
    basic_cyan = BaSiC.load_model("basic_model_cyan")
    basic_magenta = BaSiC.load_model("basic_model_magenta")
    img_tub = basic_tubulin.transform(img_tub, timelapse=True)
    img_cyan = basic_cyan.transform(img_cyan, timelapse=True)
    img_magenta = basic_magenta.transform(img_magenta, timelapse=True)

labels_1 = imread("stardist_labels_1_channel.tif")[frame]
labels_2 = imread("stardist_labels_2_channel.tif")[frame]
labels_3 = imread("stardist_labels_3_channel.tif")[frame]



assert img_stream.dims.X % tiles_per_row == 0
assert img_stream.dims.Y % tiles_per_row == 0
x_width = round(img_stream.dims.X / tiles_per_row)
y_width = round(img_stream.dims.Y / tiles_per_row)
for i in range(tiles_per_row):
    for j in range(tiles_per_row):
        x_low = i * x_width
        x_high = (i + 1) * x_width
        y_low = j * y_width
        y_high = (j + 1) * y_width

        visible = True
        if i > 0 or j > 0:
            visible = False

        cyan = viewer.add_image(img_cyan[y_low:y_high, x_low:x_high], name="cyan", colormap="cyan", blending="additive", visible=visible)
        magenta = viewer.add_image(img_magenta[y_low:y_high, x_low:x_high], name="magenta", colormap="magenta", blending="additive", visible=visible)
        tubulin = viewer.add_image(img_tub[y_low:y_high, x_low:x_high], name="tubulin", colormap="green", blending="additive", visible=visible)

        label_layer = viewer.add_labels(labels_1[y_low:y_high, x_low:x_high], visible=visible)
        label_layer = viewer.add_labels(labels_2[y_low:y_high, x_low:x_high], visible=visible)
        label_layer = viewer.add_labels(labels_3[y_low:y_high, x_low:x_high], visible=visible)
napari.run()
