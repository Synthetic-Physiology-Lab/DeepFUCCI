import json
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize

from stardist import (
    fill_label_holes,
    calculate_extents,
    gputools_available,
)
from stardist.models import Config2D, StarDist2D


training_data_dir = "../training_data"

with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

X_trn = [
    imread(f"{training_data_dir}/images/{img_name}")
    for img_name in dataset_split["training"]
]
X_val = [
    imread(f"{training_data_dir}/images/{img_name}")
    for img_name in dataset_split["validation"]
]
Y_trn = [
    fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}"))
    for img_name in dataset_split["training"]
]
Y_val = [
    fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}"))
    for img_name in dataset_split["validation"]
]

n_channel = 3

axis_norm = (0, 1)  # normalize channels independently
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

X_trn = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_trn)]
X_val = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_val)]

assert len(X_trn) == len(Y_trn)
assert len(X_val) == len(Y_val)

# 32 is a good default choice
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
)
print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used
    # by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model = StarDist2D(conf, name="stardist", basedir="models")

median_size = calculate_extents(list(Y_trn), np.median)
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
    validation_data=(X_val, Y_val),
    augmenter=augmenter,
    epochs=1000,
    steps_per_epoch=200,
)

model.optimize_thresholds(X_val, Y_val)
