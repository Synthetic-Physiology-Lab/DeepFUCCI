from aicsimageio import AICSImage
from cellpose import models
import napari
import yaml
from tqdm import tqdm
from basicpy import BaSiC
from skimage.io import imsave, imread
import numpy as np
from napari.utils import DirectLabelColormap

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

flatfield_correction = False


filename = metadata["filename"]

img_stream = AICSImage(filename)

if flatfield_correction:
    # basic_tubulin = BaSiC.load_model("basic_model_tubulin")
    basic_cyan = BaSiC.load_model("basic_model_cyan")
    basic_magenta = BaSiC.load_model("basic_model_magenta")
for T in tqdm(range(img_stream.dims.T)):
    img_tubulin = img_stream.get_image_data("YX", C=int(channels["tubulin"]), T=T)
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)
    if flatfield_correction:
        # corrected = basic_tubulin.transform(img_tubulin, timelapse=False)
        # img_tubulin = corrected.squeeze()
        corrected = basic_cyan.transform(img_cyan, timelapse=False)
        img_cyan = corrected.squeeze()
        corrected = basic_magenta.transform(img_magenta, timelapse=False)
        img_magenta = corrected.squeeze()


reference_diameter = 18
pixel_size = img_stream.physical_pixel_sizes.X
diameter = reference_diameter / pixel_size
model_green = models.CellposeModel(gpu=True, model_type="nuclei_green_v2")
model_red = models.CellposeModel(gpu=True, model_type="nuclei_red_v2")

masks_cyan, _, _ = model_green.eval(img_cyan, diameter=diameter)
masks_magenta, _, _ = model_red.eval(img_magenta, diameter=diameter)

masks = masks_cyan + masks_magenta

imsave("cellpose_masks.tif", masks, compression="zlib")
imsave("cellpose_masks_cyan.tif", masks_cyan, compression="zlib")
imsave("cellpose_masks_magenta.tif", masks_magenta, compression="zlib")

viewer = napari.Viewer()

# no crop
crop_field = [0, img_stream.dims.X]
crop_field = slice(crop_field[0], crop_field[1], 1)

img_tubulin = img_tubulin[crop_field, crop_field]
img_cyan = img_cyan[crop_field, crop_field]
img_magenta = img_magenta[crop_field, crop_field]
masks = masks[crop_field, crop_field]

tub_layer = viewer.add_image(img_tubulin, blending="additive")
cyan_layer = viewer.add_image(img_cyan, colormap="cyan", blending="additive")
magenta_layer = viewer.add_image(img_magenta, colormap="magenta", blending="additive")

tub_layer.contrast_limits = (np.percentile(img_tubulin, 1), np.percentile(img_tubulin, 99.9))
cyan_layer.contrast_limits = (np.percentile(img_cyan, 1), np.percentile(img_cyan, 99.9))
magenta_layer.contrast_limits = (np.percentile(img_magenta, 1), np.percentile(img_magenta, 99.9))
our_masks = imread("stardist_labels_3_channel.tif")
our_masks_2ch = imread("stardist_labels_2_channel.tif")
our_masks = np.squeeze(our_masks)
our_masks_2ch = np.squeeze(our_masks_2ch)
our_masks = our_masks[crop_field, crop_field]
our_masks_2ch = our_masks_2ch[crop_field, crop_field]

masks[masks > 0] = 1
label_layer = viewer.add_labels(our_masks, colormap = DirectLabelColormap(color_dict={None: "yellow"}))
label_layer_2ch = viewer.add_labels(our_masks_2ch, colormap = DirectLabelColormap(color_dict={None: "blue"}))
cp_label_layer = viewer.add_labels(masks, colormap = DirectLabelColormap(color_dict={None: "red"}))
cp_label_layer.contour = 20 
label_layer.contour = 20 
label_layer_2ch.contour = 20

napari.run()
