import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
import pyclesperanto_prototype as cle

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

X = sorted(glob("../test_data_tiled/images/*.tif"))
Y = sorted(glob("../test_data_tiled/masks/*.tif"))
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

X = [normalize(x[..., 0:2], 1, 99.8, axis=axis_norm) for x in tqdm(X)]
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

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.3, total_memory=50000)

model = StarDist2D.from_pretrained("2D_versatile_fluo")
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
    dapieq_labels, _ = model.predict_instances(max_projected)
    return dapieq_labels


Y_val_pred = [predict_instances(x) for x in tqdm(X)]


plot_img_label(X[0], Y[0], lbl_title="label GT")
plot_img_label(X[0], Y_val_pred[0], lbl_title="label Pred")

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
plt.show()

print("Stats at 0.5 IoU: ", stats[4])
