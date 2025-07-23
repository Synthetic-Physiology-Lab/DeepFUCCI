from __future__ import print_function, unicode_literals, absolute_import, division
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize
from instanseg import InstanSeg

from stardist import (
    fill_label_holes,
    random_label_cmap,
    gputools_available,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

training_data_dir = "../training_data_tiled_strict_classified_new"
# use the same data split as in training
with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

X = [imread(f"{training_data_dir}/images/{img_name}") for img_name in dataset_split["validation"]]
Y_val = [fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}")) for img_name in dataset_split["validation"]]

axis_norm = (0, 1)  # normalize channels independently
X_val = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
print("number of validation images: %3d" % len(X))

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1)
pixel_size = 0.3

Y_val_pred = [instanseg_fluorescence.eval_small_image(image=np.moveaxis(x, -1, 0), pixel_size=pixel_size, return_image_tensor=False, target="nuclei").squeeze().numpy().astype(np.uint16) for x in tqdm(X_val)]

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
plt.savefig("validation_instanseg_2ch.pdf")
plt.show()


print("Stats at 0.5 IoU: ", stats[4])
