import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

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

X = sorted(glob("training_data_tiled_strict_classified/images/*.tif"))
Y = sorted(glob("training_data_tiled_strict_classified/masks/*.tif"))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 2

axis_norm = (0, 1)  # normalize channels independently
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

X = [normalize(X[i], 1, 99.8, axis=axis_norm) for i in tqdm(ind_val)]
Y = [fill_label_holes(Y[i]) for i in tqdm(ind_val)]

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


# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model_1d = StarDist2D(None, name="stardist", basedir="training_1_channel_stardist/models")
model_2d = StarDist2D(None, name="stardist", basedir="training_2_channels_stardist//models")
model_3d = StarDist2D(None, name="stardist", basedir="training_3_channels_stardist//models")

Y_val_pred_1d = [
    model_1d.predict_instances(
        x[..., 2]
    )[0]
    for x in tqdm(X)
]

Y_val_pred_2d = [
    model_2d.predict_instances(
        x[..., 0:2]
    )[0]
    for x in tqdm(X)
]

Y_val_pred_3d = [
    model_3d.predict_instances(
        x
    )[0]
    for x in tqdm(X)
]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats_1d = [
    matching_dataset(Y, Y_val_pred_1d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

stats_2d = [
    matching_dataset(Y, Y_val_pred_2d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

stats_3d = [
    matching_dataset(Y, Y_val_pred_3d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]


for m in (
    "accuracy",
    "f1",
):
    plt.plot(taus, [s._asdict()[m] for s in stats_1d], ".-", lw=2, label="1D")
    plt.plot(taus, [s._asdict()[m] for s in stats_2d], ".-", lw=2, label="2D")
    plt.plot(taus, [s._asdict()[m] for s in stats_3d], ".-", lw=2, label="3D")
    plt.xlabel(r"IoU threshold $\tau$")
    plt.ylabel(f"{m.capitalize()} value")
    plt.grid()
    plt.legend()

    plt.savefig(f"{m}.pdf")
    plt.show()

print("Stats at 0.5 IoU: ", stats_1d[4])
print("Stats at 0.5 IoU: ", stats_2d[4])
print("Stats at 0.5 IoU: ", stats_3d[4])
