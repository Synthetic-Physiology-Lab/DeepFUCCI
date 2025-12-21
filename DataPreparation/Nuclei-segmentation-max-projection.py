import os
from basicpy import BaSiC
from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
import numpy as np
import tifffile
import yaml
from tqdm import tqdm
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import napari

size_ratio_filter = 0.5

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)
filename = metadata["filename"]


nucleus_radius = 0.5 * float(metadata["nucleus_diameter"])

# Channels
# The original images have four channels.
# They are labelled here to make the code later more comprehensible.
# note: zero-based
channels = metadata["channels"]
print("Channels: ", channels)

flatfield_correction = True
for channel in channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")

# Read the image.
# This is one of the variables of the entire workflow: the filename.

image = AICSImage(filename)
number_of_frames = image.shape[0]

pixel_size = image.physical_pixel_sizes.X
nucleus_radius_pixel = nucleus_radius / pixel_size

if flatfield_correction:
    basic_cyan = BaSiC.load_model("basic_model_cyan")
    basic_magenta = BaSiC.load_model("basic_model_magenta")

# Load data from image for current frame.
def load_channels(current_frame: int):
    # shading correction
    magenta = image.get_image_data("YX", T=current_frame, C=int(channels["magenta"]))
    cyan = image.get_image_data("YX", T=current_frame, C=int(channels["cyan"]))
    if flatfield_correction:
        corrected = basic_cyan.transform(cyan, timelapse=False)
        cyan = corrected.squeeze()
        corrected = basic_magenta.transform(magenta, timelapse=False)
        magenta = corrected.squeeze()

    return magenta, cyan


dapieq_labels = np.zeros((number_of_frames, image.shape[3], image.shape[4]), dtype=np.uint16)
merged_frames = np.zeros((number_of_frames, image.shape[3], image.shape[4]))
magenta_frames = np.zeros((number_of_frames, image.shape[3], image.shape[4]))
cyan_frames = np.zeros((number_of_frames, image.shape[3], image.shape[4]))
# stardist
model = StarDist2D.from_pretrained('2D_versatile_fluo')

for current_frame in tqdm(range(number_of_frames)):
    magenta, cyan = load_channels(current_frame)

    # remove background
    magenta_top = cle.top_hat_sphere(magenta, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel)
    magenta_frames[current_frame, :] = magenta_top[:]
    # blur
    magenta_blur = cle.gaussian_blur(magenta_top, sigma_x=2.0, sigma_y=2.0)
    normal_magenta = normalize(magenta_blur.get())

    cyan_top = cle.top_hat_sphere(cyan, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel)
    cyan_frames[current_frame, :] = cyan_top[:]
    cyan_blur = cle.gaussian_blur(cyan_top, sigma_x=2.0, sigma_y=2.0)
    normal_cyan = normalize(cyan_blur.get())

    max_projected = np.maximum(normal_magenta, normal_cyan)
    merged_frames[current_frame, :] = max_projected[:]

    tmp_dapieq_labels, _ = model.predict_instances(max_projected)
    # filter small labels
    tmp_dapieq_labels = cle.exclude_small_labels(tmp_dapieq_labels,
                                                 maximum_size=np.pi * (size_ratio_filter * nucleus_radius_pixel)**2)

    # exclude labels on edges
    tmp_dapieq_labels = cle.exclude_labels_on_edges(tmp_dapieq_labels)

    # copy labels
    dapieq_labels[current_frame, :] = tmp_dapieq_labels[:]

tifffile.imwrite("dapieq_labels.tif", dapieq_labels, dtype=np.uint16, compression="zlib")
tifffile.imwrite("merged_frames.tif", merged_frames, compression="zlib")

viewer = napari.Viewer()
viewer.add_image(magenta_frames, name="magenta", colormap="magenta", blending="additive")
viewer.add_image(cyan_frames, name="cyan", colormap="cyan", blending="additive")
viewer.add_image(merged_frames, name="merged", blending="additive")
viewer.add_labels(dapieq_labels)
napari.run()
