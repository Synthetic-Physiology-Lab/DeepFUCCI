import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from cellpose import denoise
from pathlib import Path

from stardist import (
    fill_label_holes,
    random_label_cmap,
)
from stardist.matching import matching_dataset

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

data_file = "image_cyan_magenta_last_frame.tif"
if not Path(data_file).exists():
    data_file = Path("../../../../data/test_cellmaptracer") / data_file
    if not Path(data_file).exists():
        raise FileNotFoundError("Data file not there")


X_val = [np.moveaxis(imread(data_file), 0, -1)]
Y_val = [fill_label_holes(imread("gt_last_frame.tif"))]

# Use CellposeDenoiseModel with cyto3 and denoise_cyto3 restoration
model = denoise.CellposeDenoiseModel(
    gpu=True, model_type="cyto3", restore_type="denoise_cyto3", chan2_restore=True
)

nucleus_radius_pixel = 10 / 0.3  # 10 microns divided by 0.3 microns per pixel


def predict_instances(x):
    ch1 = x[..., 0]
    ch2 = x[..., 1]

    max_projected = np.maximum(ch1, ch2)
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
