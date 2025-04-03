# Following the example here: https://github.com/stardist/stardist/blob/main/examples/other2D/multiclass.ipynb


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
import json

from stardist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from stardist.matching import matching_dataset
from stardist.models import Config2D, StarDist2D
from stardist.utils import mask_to_categorical
from stardist.plot import render_label


# show plots?
show = False

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()
lbl_cmap_classes = matplotlib.cm.tab20

# Attention: insert the directory here!
training_data_dir = None
if training_data_dir is None:
    raise ValueError("Provide a directory where the training data is located")

X = sorted(glob(f"{training_data_dir}/images/*.tif"))
Y = sorted(glob(f"{training_data_dir}/masks/*.tif"))
C = sorted(glob(f"{training_data_dir}/classes/*.json"))

assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))
assert all(
    Path(x).name.replace(".json", "") == Path(y).name.replace(".tif", "")
    for x, y in zip(C, Y)
)


def read_json(path):
    with open(path) as fp:
        data = json.load(fp)
    data_list = []
    for key, value in sorted(data.items()):
        data_list.append((int(key), value))

    return dict(data_list)


X = list(map(imread, X))
Y = list(map(imread, Y))
C = list(map(read_json, C))

n_channel = 2
n_classes = 3

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
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val, C_val = (
    [X[i] for i in ind_val],
    [Y[i] for i in ind_val],
    [C[i] for i in ind_val],
)
X_trn, Y_trn, C_trn = (
    [X[i] for i in ind_train],
    [Y[i] for i in ind_train],
    [C[i] for i in ind_train],
)
print(X[0].shape)
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))


def plot_img_label(
    img,
    lbl,
    cls_dict,
    n_classes=2,
    img_title="image",
    lbl_title="label",
    cls_title="classes",
    **kwargs,
):
    c = mask_to_categorical(lbl, n_classes=n_classes, classes=cls_dict)
    res = np.zeros(lbl.shape, np.uint16)
    for i in range(1, c.shape[-1]):
        m = c[..., i] > 0
        res[m] = i
    class_img = lbl_cmap_classes(res)
    class_img[..., :3][res == 0] = 0
    class_img[..., -1][res == 0] = 1

    fig, (ai, al, ac) = plt.subplots(
        1, 3, figsize=(17, 7), gridspec_kw=dict(width_ratios=(1.0, 1, 1))
    )

    if len(img.shape) == 3 and img.shape[2] == 2:
        print(img.shape)
        img = np.concatenate(
            (img, np.zeros(shape=(img.shape[0], img.shape[1], 1))), axis=-1
        )
        print(img.shape)

    _ = ai.imshow(img, cmap="gray")
    # fig.colorbar(im, ax = ai)
    ai.set_title(img_title)
    al.imshow(
        render_label(
            lbl,
            0.8 * normalize(img, clip=True),
            normalize_img=False,
            alpha_boundary=0.8,
            cmap=lbl_cmap,
        )
    )
    al.set_title(lbl_title)
    ac.imshow(class_img)
    ac.imshow(
        render_label(
            res,
            0.8 * normalize(img, clip=True),
            normalize_img=False,
            alpha_boundary=0.8,
            cmap=lbl_cmap_classes,
        )
    )
    ac.set_title(cls_title)
    plt.tight_layout()
    for a in ai, al, ac:
        a.axis("off")
    return ai, al, ac


i = min(8, len(X) - 1)
img, lbl, cls = X[i], Y[i], C[i]
assert img.ndim in (2, 3)
img = img if (img.ndim == 2 or img.shape[-1] == 3) else img[..., 0]
plot_img_label(img, lbl, cls, n_classes=n_classes)
if show:
    plt.show()
else:
    plt.savefig("example.png")
    plt.close()

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2, 2)

conf = Config2D(
    n_rays=n_rays,
    grid=grid,
    use_gpu=use_gpu,
    n_channel_in=n_channel,
    n_classes=n_classes,
)
print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.3, total_memory=50000)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)

model = StarDist2D(conf, name="stardist_multiclass", basedir="models")

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap("YX"))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print(
        "WARNING: median object size larger than field of view of the neural network."
    )


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


model.train(
    X_trn,
    Y_trn,
    classes=C_trn,
    validation_data=(X_val, Y_val, C_val),
    augmenter=augmenter,
    epochs=1000,
    steps_per_epoch=200,
)

model.optimize_thresholds(X_val, Y_val)

i = 8
label, res = model.predict_instances(X_val[i], n_tiles=model._guess_n_tiles(X_val[i]))

# the class object ids are stored in the 'results' dict and correspond to the label ids in increasing order


def class_from_res(res):
    cls_dict = dict((i + 1, c) for i, c in enumerate(res["class_id"]))
    return cls_dict


print(class_from_res(res))

plot_img_label(X_val[i], Y_val[i], C_val[i], lbl_title="GT", n_classes=n_classes)
plot_img_label(
    X_val[i], label, class_from_res(res), lbl_title="Pred", n_classes=n_classes
)
if show:
    plt.show()
else:
    plt.savefig("prediction_8.png")
    plt.close()


Y_val_pred, res_val_pred = tuple(
    zip(
        *[
            model.predict_instances(
                x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False
            )
            for x in tqdm(X_val[:])
        ]
    )
)

plot_img_label(X_val[0], Y_val[0], C_val[0], lbl_title="label GT", n_classes=n_classes)
plot_img_label(
    X_val[0],
    Y_val_pred[0],
    class_from_res(res_val_pred[0]),
    lbl_title="label Pred",
    n_classes=n_classes,
)
if show:
    plt.show()
else:
    plt.savefig("prediction_0.png")
    plt.close()

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
if show:
    plt.show()
else:
    plt.savefig("metrics.pdf")
    plt.close()
