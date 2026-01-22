from __future__ import print_function, unicode_literals, absolute_import, division
from glob import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import label as skimage_label
from skimage.io import imread
from csbdeep.utils import Path, normalize
import pyclesperanto_prototype as cle
from cellpose import models

from stardist import (
    fill_label_holes,
    random_label_cmap,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

DATA_DIR = "../../../data"
test_data_dir = f"{DATA_DIR}/test_confluent_fucci_data"
X = sorted(glob(f"{test_data_dir}/images/*.tif"))
Y = sorted(glob(f"{test_data_dir}/masks/*.tif"))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))

X_val = [x[..., 0:2] for x in tqdm(X)]
Y_val = [fill_label_holes(skimage_label(y)) for y in tqdm(Y)]
print("number of validation images: %3d" % len(X))


model = models.Cellpose(model_type="cyto3")
nucleus_radius_pixel = 10 / 0.3  # 10 microns divided by 0.3 microns per pixel


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


Y_val_pred = [predict_instances(x) for x in tqdm(X_val)]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [
    matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False)
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
plt.savefig("validation_dapieq_cellpose_confluent.pdf")
plt.show()

print("Stats at 0.5 IoU: ", stats[4])
