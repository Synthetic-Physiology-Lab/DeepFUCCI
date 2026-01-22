import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from csbdeep.utils import normalize
from aicsimageio import AICSImage
from pathlib import Path

from stardist import (
    fill_label_holes,
    random_label_cmap,
    gputools_available,
)
from stardist.matching import matching_dataset
from stardist.models import StarDist2D

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

filename = metadata["filename"]
if not Path(filename).exists():
    filename = Path("../../../data/HaCaT_Han_et_al") / filename


img_stream = AICSImage(filename)
label_stream = AICSImage("labels_manual_annotation.tif")

X = []
Y = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_1d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_2d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_3d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)


model_1d = StarDist2D(
    None, name="stardist_1_channel_latest", basedir=Path.home() / "models"
)
model_2d = StarDist2D(
    None, name="stardist_2_channel_latest", basedir=Path.home() / "models"
)
model_3d = StarDist2D(
    None, name="stardist_3_channel_latest", basedir=Path.home() / "models"
)


for T in tqdm(range(img_stream.dims.T)):
    img_tubulin = img_stream.get_image_data("YX", C=int(channels["tubulin"]), T=T)
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)
    gt_labels = label_stream.get_image_data("YX", Z=T)

    # normalize image
    img_tubulin = normalize(img_tubulin, pmin=1, clip=True)
    img_cyan = normalize(img_cyan, pmin=1, clip=True)
    img_magenta = normalize(img_magenta, pmin=1, clip=True)

    X.append(img_tubulin)
    # 1 channel model
    labels, details = model_1d.predict_instances(img_tubulin)
    labels_1d[T] = labels[:]

    # 2 channel model
    labels, details = model_2d.predict_instances(
        np.moveaxis(np.stack([img_cyan, img_magenta]), 0, -1)
    )
    labels_2d[T] = labels[:]

    # 3 channel model
    labels, details = model_3d.predict_instances(
        np.moveaxis(np.stack([img_cyan, img_magenta, img_tubulin]), 0, -1)
    )
    labels_3d[T] = labels[:]
    Y[T] = gt_labels[:]

Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
print("number of images: %3d" % len(X))


def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai, al) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1))
    )
    if len(img.shape) == 3 and img.shape[2] == 2:
        print(img.shape)
        img = np.concatenate(
            (img, np.zeros(shape=(img.shape[0], img.shape[1], 1))), axis=-1
        )
        print(img.shape)

    im = ai.imshow(img, cmap="gray", clim=(0, 1))

    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


i = min(9, len(X) - 1)
img, lbl = X[i], Y[i]
assert img.ndim in (2, 3)
img = img if (img.ndim == 2 or img.shape[-1] == 3) else img[..., 0]
plot_img_label(img, lbl)
plt.show()

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

i = 1
for Y_val_pred in [labels_1d, labels_2d, labels_3d]:
    plot_img_label(X[0], Y[0], lbl_title="label GT")
    plot_img_label(X[0], Y_val_pred[0], lbl_title="label Pred")

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
    plt.show()

    print("Stats at 0.5 IoU for {i} CH: ", stats[4])
    i += 1
