import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from csbdeep.utils import Path, normalize
from cellpose import models
import pyclesperanto_prototype as cle
from glob import glob


from stardist import (
    random_label_cmap,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

DATA_DIR = "../../../data"
test_data_dir = f"{DATA_DIR}/data_set_HT1080_40x"
X = sorted(glob(f"{test_data_dir}/images/*.tif"))
Y = sorted(glob(f"{test_data_dir}/masks/*.tif"))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 2

axis_norm = (0, 1)  # normalize channels independently
X_val = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
print("number of validation images: %3d" % len(X))

reference_diameter = 18
pixel_size = 0.33
diameter = reference_diameter / pixel_size
model_green = models.CellposeModel(gpu=True, model_type="nuclei_green_v2")
model_red = models.CellposeModel(gpu=True, model_type="nuclei_red_v2")


def get_confluentfucci_masks(img_cyan, img_magenta):
    masks_cyan, _, _ = model_green.eval(img_cyan, diameter=diameter)
    masks_magenta, _, _ = model_red.eval(img_magenta, diameter=diameter)
    # refine masks
    masks_cyan = cle.opening_labels(masks_cyan, radius=3)
    masks_magenta = cle.opening_labels(masks_magenta, radius=3)

    masks = cle.merge_touching_labels(masks_cyan + masks_magenta)
    masks = cle.closing_labels(cle.label(masks), radius=3).get()
    return masks


Y_val_pred = [get_confluentfucci_masks(x[..., 1], x[..., 0]) for x in tqdm(X_val)]

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
plt.savefig("test_confluentfucci.pdf")
plt.show()


print("Stats at 0.5 IoU: ", stats[4])
