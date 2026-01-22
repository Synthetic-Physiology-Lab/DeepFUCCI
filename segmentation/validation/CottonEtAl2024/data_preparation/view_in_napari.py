from aicsimageio import AICSImage
from skimage.io import imread
import napari

viewer = napari.Viewer()

filename = "MJC004_Pos1_Control_4C_FUCCI.tif"
img_stream = AICSImage(filename)

img_cyan = img_stream.get_image_dask_data("TYX", C=0)
img_magenta = img_stream.get_image_dask_data("TYX", C=1)

cyan_layer = viewer.add_image(
    img_cyan, colormap="cyan", blending="additive", name="cyan"
)
magenta_layer = viewer.add_image(
    img_magenta, colormap="magenta", blending="additive", name="magenta"
)

labels_2 = imread("stardist_labels_2_channel.tif")
labels_gt = imread("cellpose_masks.tif")

viewer.add_labels(labels_2, scale=cyan_layer.scale)
viewer.add_labels(labels_gt, scale=cyan_layer.scale)

napari.run()
