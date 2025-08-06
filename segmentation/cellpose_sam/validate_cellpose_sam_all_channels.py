import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from cellpose import models

from stardist import (
    fill_label_holes,
    gputools_available,
)
from stardist.matching import matching_dataset

training_data_dir = "../training_data_tiled_strict_classified_new"
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

print("number of validation images: %3d" % len(X))

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

model = models.CellposeModel(gpu=True, pretrained_model="models/FUCCI_cpsam")

Y_val_pred_1d = []
Y_val_pred_2d = []
Y_val_pred_3d = []
for x in tqdm(X):
    masks, _, _ = model.eval(x[..., 2], batch_size=32)
    Y_val_pred_1d.append(masks)
    masks, _, _ = model.eval(x[..., 0:2], batch_size=32)
    Y_val_pred_2d.append(masks)
    masks, _, _ = model.eval(x, batch_size=32)
    Y_val_pred_3d.append(masks)


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
    plt.plot(
        taus,
        [s._asdict()[m] for s in stats_1d],
        ls="dashdot",
        lw=2,
        label="1 CH",
        color="black",
    )
    plt.plot(
        taus,
        [s._asdict()[m] for s in stats_2d],
        ls="dotted",
        lw=2,
        label="2 CH",
        color="black",
    )
    plt.plot(
        taus,
        [s._asdict()[m] for s in stats_3d],
        ls="dashed",
        lw=2,
        label="3 CH",
        color="black",
    )
    plt.xlabel("IoU threshold")
    plt.ylabel(f"{m.capitalize()} value")
    plt.grid()
    plt.legend()
    plt.ylim(0, 1.05)

    plt.savefig(f"validation_cellpose_sam_{m}.pdf")
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

plt.savefig("validation_cellpose_sam_label_numbers.pdf")
plt.show()

print("Stats at 0.5 IoU for 1 CH: ", stats_1d[4])
print("Stats at 0.5 IoU for 2 CH: ", stats_2d[4])
print("Stats at 0.5 IoU for 3 CH: ", stats_3d[4])
