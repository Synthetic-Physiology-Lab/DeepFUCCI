import pyclesperanto_prototype as cle
from aicsimageio import AICSImage
from cellpose import models
import napari
import yaml
from tqdm import tqdm
from basicpy import BaSiC
from skimage.io import imsave, imread
import numpy as np
from napari.utils import DirectLabelColormap
from skimage.measure import regionprops_table
import pandas as pd
from instanseg import InstanSeg
import os

os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.expanduser("~/Documents/github/instanseg/instanseg/bioimageio_models")
instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1)

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[..., 0]
    minc = bbox_extents[..., 1]
    maxr = bbox_extents[..., 2]
    maxc = bbox_extents[..., 3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect



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

masks = cle.merge_touching_labels(masks_cyan + masks_magenta)
masks = cle.label(masks).get()


instanseg_labels = instanseg_fluorescence.eval_small_image(image=np.stack([img_cyan, img_magenta, img_tubulin]), pixel_size=pixel_size, return_image_tensor=False, target="nuclei")

imsave("cellpose_masks.tif", masks, compression="zlib")
imsave("cellpose_masks_cyan.tif", masks_cyan, compression="zlib")
imsave("cellpose_masks_magenta.tif", masks_magenta, compression="zlib")
imsave("instanseg_masks.tif", masks, compression="zlib")

viewer = napari.Viewer()

# no crop
crop_field = [0, img_stream.dims.X]
crop_field = slice(crop_field[0], crop_field[1], 1)

img_tubulin = img_tubulin[crop_field, crop_field]
img_cyan = img_cyan[crop_field, crop_field]
img_magenta = img_magenta[crop_field, crop_field]
masks = masks[crop_field, crop_field]

viewer.add_labels(masks, name="test")
tub_layer = viewer.add_image(img_tubulin, blending="additive")
cyan_layer = viewer.add_image(img_cyan, colormap="cyan", blending="additive")
magenta_layer = viewer.add_image(img_magenta, colormap="magenta", blending="additive")

tub_layer.contrast_limits = (np.percentile(img_tubulin, 1), np.percentile(img_tubulin, 99.9))
cyan_layer.contrast_limits = (np.percentile(img_cyan, 1), np.percentile(img_cyan, 99.9))
magenta_layer.contrast_limits = (np.percentile(img_magenta, 1), np.percentile(img_magenta, 99.9))
our_masks = imread("stardist_labels_3_channel.tif")
our_masks_2ch = imread("stardist_labels_2_channel.tif")
our_masks_1ch = imread("stardist_labels_1_channel.tif")
our_masks = np.squeeze(our_masks)
our_masks_2ch = np.squeeze(our_masks_2ch)
our_masks_1ch = np.squeeze(our_masks_1ch)
our_masks = our_masks[crop_field, crop_field]
our_masks_2ch = our_masks_2ch[crop_field, crop_field]
our_masks_1ch = our_masks_1ch[crop_field, crop_field]
# convert from pytorch
instanseg_labels = instanseg_labels.detach().numpy().astype(np.int32).squeeze()
instanseg_labels = instanseg_labels[crop_field, crop_field]


# create bounding boxes
def add_bounding_boxes(masks, name, color, edge_width=4):
    props = regionprops_table(masks, properties=("label", "centroid", "bbox"))
    data = pd.DataFrame(props)

    bbox_rects = []
    for _, row in data.iterrows():
        bbox_rects.append([row[f'bbox-{i}'] for i in range(4)])

    bbox_rects = np.array(bbox_rects)
    bbox_rects = make_bbox(bbox_rects)

    _ = viewer.add_shapes(
        bbox_rects,
        face_color='transparent',
        edge_color=color,
        name=name,
        edge_width=edge_width
    )

add_bounding_boxes(masks, name="bboxes_confluent", color="red", edge_width=20)
add_bounding_boxes(instanseg_labels, name="bboxes_instanseg", color="green", edge_width=20)

new_masks = masks.copy()
new_masks[masks > 0] = 1
label_layer = viewer.add_labels(our_masks, colormap = DirectLabelColormap(color_dict={None: "yellow"}))
label_layer_2ch = viewer.add_labels(our_masks_2ch, colormap = DirectLabelColormap(color_dict={None: "yellow"}))
label_layer_1ch = viewer.add_labels(our_masks_1ch, colormap = DirectLabelColormap(color_dict={None: "yellow"}))
cp_label_layer = viewer.add_labels(new_masks, colormap = DirectLabelColormap(color_dict={None: "red"}))
cp_label_layer.contour = 20 
label_layer.contour = 20 
label_layer_2ch.contour = 20

instanseg_label_layer = viewer.add_labels(instanseg_labels, colormap = DirectLabelColormap(color_dict={None: "green"}))
instanseg_label_layer.visible = False

napari.run()
