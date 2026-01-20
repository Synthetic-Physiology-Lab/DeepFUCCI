import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from skimage.measure import label as label_skimage
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

DATA_DIR = "../../data"
test_data_dir = f"{DATA_DIR}/data_set_HT1080_100x"
X = sorted(glob(f"{test_data_dir}/images/*.tif"))
Y = sorted(glob(f"{test_data_dir}/masks/*.tif"))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))

axis_norm = (0, 1)  # normalize channels independently
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(label_skimage(y)) for y in tqdm(Y)]
print("number of test images: %3d" % len(X))

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model_1d = StarDist2D(None, name="stardist_1_channel_latest", basedir=Path.home() / "models")
model_2d = StarDist2D(None, name="stardist_2_channel_latest", basedir=Path.home() / "models")
model_3d = StarDist2D(None, name="stardist_3_channel_latest", basedir=Path.home() / "models")

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
    plt.clf()
    plt.plot(taus, [s._asdict()[m] for s in stats_1d], ls="dashdot", lw=2, label="1 CH", color="black")
    plt.plot(taus, [s._asdict()[m] for s in stats_2d], ls="dotted", lw=2, label="2 CH", color="black")
    plt.plot(taus, [s._asdict()[m] for s in stats_3d], ls="dashed", lw=2, label="3 CH", color="black")
    plt.xlabel("IoU threshold")
    plt.ylabel(f"{m.capitalize()} value")
    plt.grid()
    plt.legend()
    plt.ylim(0, 1.05)

    plt.savefig(f"test{m}.pdf")
    plt.show()

plt.clf()
for idx, stats in enumerate([stats_1d, stats_2d, stats_3d]):
    linestyle = ["dashdot", "dotted", "dashed"][idx]
    for index, m in enumerate(["fp", "tp", "fn"]):
        label = None
        if idx == 0:
            label = m.upper()
        
        plt.plot(taus, [s._asdict()[m] for s in stats], ls=linestyle, lw=2, label=label)
    plt.gca().set_prop_cycle(None)
plt.xlabel("IoU threshold")
plt.ylabel("Number of labels")
plt.grid()
plt.legend()

plt.savefig("test_label_numbers.pdf")
plt.show()

print("Stats at 0.5 IoU for 1 CH: ", stats_1d[4])
print("Stats at 0.5 IoU for 2 CH: ", stats_2d[4])
print("Stats at 0.5 IoU for 3 CH: ", stats_3d[4])
