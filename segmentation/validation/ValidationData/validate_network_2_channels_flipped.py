import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize
import json

from stardist import (
    fill_label_holes,
    gputools_available,
)
from stardist.matching import matching_dataset
from stardist.models import StarDist2D
from pathlib import Path

DATA_DIR = "../../../data"
training_data_dir = f"{DATA_DIR}/training_data_tiled_strict_classified"
# use the same data split as in training
with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

X = [
    imread(f"{training_data_dir}/images/{img_name}")
    for img_name in dataset_split["validation"]
]
Y = [
    fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}"))
    for img_name in dataset_split["validation"]
]

axis_norm = (0, 1)  # normalize channels independently
X = [normalize(np.flip(x[..., 0:2], -1), 1, 99.8, axis=axis_norm) for x in tqdm(X)]
print("number of validation images: %3d" % len(X))

assert len(X) > 1, "not enough training data"
print("number of images: %3d" % len(X))


# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory

    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model = StarDist2D(
    None, name="stardist_2_channel_latest", basedir=Path.home() / "models"
)

Y_val_pred = [model.predict_instances(x)[0] for x in tqdm(X)]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [
    matching_dataset(Y, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)
]

for m in (
    "accuracy",
    "f1",
):
    plt.clf()
    plt.plot(
        taus,
        [s._asdict()[m] for s in stats],
        ls="dotted",
        lw=2,
        label="2 CH swapped",
        color="black",
    )
    plt.xlabel("IoU threshold")
    plt.ylabel(f"{m.capitalize()} value")
    plt.grid()
    plt.legend()
    plt.ylim(0, 1.05)

    plt.savefig(f"validation_2CH_swapped_channels_{m}.pdf")
    plt.show()

plt.clf()
for m in ("fp", "tp", "fn"):
    label = m.upper()
    linestyle = "dotted"
    plt.plot(taus, [s._asdict()[m] for s in stats], ls=linestyle, lw=2, label=label)

    plt.gca().set_prop_cycle(None)
plt.xlabel("IoU threshold")
plt.ylabel("Number of labels")
plt.grid()
plt.legend()

plt.savefig("validation_2CH_swapped_label_numbers.pdf")
plt.show()

print("Stats at 0.5 IoU: ", stats[4])
