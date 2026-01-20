from __future__ import print_function, unicode_literals, absolute_import, division
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize
from cellpose import denoise

from stardist import (
    fill_label_holes,
    random_label_cmap,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

DATA_DIR = "../../data"
training_data_dir = f"{DATA_DIR}/training_data_tiled_strict_classified"
# use the same data split as in training
with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

X = [imread(f"{training_data_dir}/images/{img_name}") for img_name in dataset_split["validation"]]
Y_val = [fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}")) for img_name in dataset_split["validation"]]

axis_norm = (0, 1)  # normalize channels independently
X_val = [normalize(x[..., 0:2], 1, 99.8, axis=axis_norm) for x in tqdm(X)]
print("number of validation images: %3d" % len(X))

# Use CellposeDenoiseModel with cyto3 and denoise_cyto3 restoration
model = denoise.CellposeDenoiseModel(
    gpu=True,
    model_type="cyto3",
    restore_type="denoise_cyto3",
    chan2_restore=True
)
nucleus_radius_pixel = 10 / 0.3  # 10 microns divided by 0.3 microns per pixel


def predict_instances(x):
    ch1 = x[..., 0]
    ch2 = x[..., 1]

    # Create a max projection of both channels for nuclear segmentation
    max_projected = np.maximum(ch1, ch2)

    # Stack as 2D grayscale image for cyto3 with denoising
    # channels=[0,0] means grayscale image with no secondary channel
    masks, flows, styles, imgs_restored = model.eval(
        max_projected,
        diameter=2.0 * nucleus_radius_pixel,
        channels=[0, 0]
    )
    return masks


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
plt.savefig("validation_dapieq_cellpose_cyto3_denoise.pdf")
plt.show()

print("Stats at 0.5 IoU: ", stats[4])
