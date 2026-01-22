import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from csbdeep.utils import normalize
from bioio import BioImage
import bioio_ome_tiff
import bioio_tifffile
import pyclesperanto_prototype as cle
from cellpose import models
from pathlib import Path

from stardist import (
    fill_label_holes,
)
from stardist.matching import matching_dataset

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

filename = metadata["filename"]
if not Path(filename).exists():
    filename = Path("../../../data/HaCaT_Han_et_al") / filename


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


model = models.Cellpose(model_type="cyto3")
nucleus_radius_pixel = 10 / 0.33  # 10 microns divided by 0.3 microns per pixel


def predict_instances(x):
    ch1 = x[..., 0]
    ch2 = x[..., 1]

    ch1_top = cle.top_hat_sphere(
        ch1, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel
    )
    # blur
    ch1_blur = cle.gaussian_blur(ch1_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch1 = normalize(ch1_blur.get())

    ch2_top = cle.top_hat_sphere(
        ch2, radius_x=2.0 * nucleus_radius_pixel, radius_y=2.0 * nucleus_radius_pixel
    )
    ch2_blur = cle.gaussian_blur(ch2_top, sigma_x=2.0, sigma_y=2.0)
    normal_ch2 = normalize(ch2_blur.get())

    max_projected = np.maximum(normal_ch1, normal_ch2)
    channels = [[0, 0]]
    dapieq_labels, flows, styles, diams = model.eval(
        max_projected, diameter=2.0 * nucleus_radius_pixel, channels=channels
    )
    return dapieq_labels


Y_val_pred = [predict_instances(x) for x in tqdm(X)]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [
    matching_dataset(Y, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)
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
plt.savefig("validation_dapieq_cellpose.pdf")
plt.show()

print("Stats at 0.5 IoU: ", stats[4])
