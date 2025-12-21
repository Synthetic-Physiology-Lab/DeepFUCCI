import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imsave
from bioio import BioImage
import bioio_ome_tiff
import bioio_tifffile
from cellpose import models
import pyclesperanto_prototype as cle

from stardist import (
    fill_label_holes,
    random_label_cmap,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

filename = "merged.ome.tif"
metadata["filename"]

img_stream = BioImage(filename, reader=bioio_ome_tiff.Reader)
label_stream = BioImage("labels_manual_annotation.tif", reader=bioio_tifffile.Reader)

X = []
Y = []

for T in tqdm(range(img_stream.dims.T)):
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)
    gt_labels = label_stream.get_image_data("YX", Z=T) 
    # normalize image
    Y.append(gt_labels)
    X.append(np.moveaxis(np.stack([img_cyan, img_magenta]), 0, -1))

Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
print("number of images: %3d" % len(X))


reference_diameter = 18
pixel_size = 0.33
diameter = reference_diameter / pixel_size
model_green = models.CellposeModel(gpu=True, model_type="nuclei_green_v2")
model_red = models.CellposeModel(gpu=True, model_type="nuclei_red_v2")

def get_confluentfucci_masks(img_cyan, img_magenta):
    masks_cyan, _, _ = model_green.eval(img_cyan, diameter=diameter)
    masks_magenta, _, _ = model_red.eval(img_magenta, diameter=diameter)
    # refine masks
    masks_cyan = cle.opening_labels(masks_cyan,
                                    radius=3)
    masks_magenta = cle.opening_labels(masks_magenta,
                                    radius=3)

    masks = cle.merge_touching_labels(masks_cyan + masks_magenta)
    masks = cle.closing_labels(cle.label(masks), radius=3).get()
    return masks


print(X[0].shape, Y[0].shape)
Y_val_pred = [get_confluentfucci_masks(x[..., 0], x[..., 1]) for x in tqdm(X)]
imsave("predicted_labels_confluentfucci.tif", Y_val_pred[0], compression="zlib")

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [
    matching_dataset(Y, Y_val_pred, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for m in (
    "precision",
    "recall",
    "accuracy",
    "f1",
    "mean_true_score",
    "mean_matched_score",
    "panoptic_quality",
):
    ax1.plot(taus, [s._asdict()[m] for s in stats], ".-", lw=2, label=m)
ax1.set_xlabel(r"IoU threshold $\tau$")
ax1.set_ylabel("Metric value")
ax1.grid()
ax1.legend()

for m in ("fp", "tp", "fn"):
    ax2.plot(taus, [s._asdict()[m] for s in stats], ".-", lw=2, label=m)
ax2.set_xlabel(r"IoU threshold $\tau$")
ax2.set_ylabel("Number #")
ax2.grid()
ax2.legend()
plt.savefig("validation_confluentfucci.pdf")
plt.show()


print("Stats at 0.5 IoU: ", stats[4])
