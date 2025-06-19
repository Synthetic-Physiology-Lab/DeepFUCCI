import pandas as pd
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fucciphase.phase import estimate_cell_phase_from_max_intensity
from fucciphase.sensor import get_fuccisa_default_sensor
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from skimage.util import invert
from pathlib import Path

from stardist import (
    fill_label_holes,
    random_label_cmap,
    gputools_available,
)
from stardist.matching import matching_dataset, group_matching_labels
from stardist.models import StarDist2D

matplotlib.rcParams["image.interpolation"] = "none"
np.random.seed(42)
lbl_cmap = random_label_cmap()


def class_from_res(res):
    # cast to int (for json)
    cls_dict = dict((int(i+1),int(c)) for i,c in enumerate(res['class_id']))
    return cls_dict

def convert_categories(category: str) -> int:
    if category == "G1":
        return 1
    elif category == "G1/S":
        return 2
    elif category == "S/G2/M":
        return 3

training_data_dir = "training_data_tiled_strict_classified_new"
# use the same data split as in training
with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

X = [f"{training_data_dir}/images/{img_name}" for img_name in dataset_split["validation"]]
Y = [f"{training_data_dir}/masks/{img_name}" for img_name in dataset_split["validation"]]
C = [f"{training_data_dir}/classes/{img_name.replace('.tif', '.json')}" for img_name in dataset_split["validation"]]
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

# comment later
X = X[:5]
Y = Y[:5]
C = C[:5]

X = list(map(imread, X))
Y = list(map(imread, Y))

# load and normalize images
axis_norm = (0, 1)  # normalize channels independently
assert X[0].shape[2] == 3
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough validation data"
print("number of images: %3d" % len(X))

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model_1d = StarDist2D(None, name="stardist_multiclass", basedir="training_1_channel_stardist_classifier/models")
model_2d = StarDist2D(None, name="stardist_multiclass", basedir="training_2_channels_stardist_classifier/models")
model_3d = StarDist2D(None, name="stardist_multiclass", basedir="training_3_channels_stardist_classifier/models")

labels_1d = []
classes_1d = []
labels_2d = []
classes_2d = []
labels_3d = []
classes_3d = []
classes_fucciphase = []

channels = ["mean_intensity-0", "mean_intensity-1"]
sensor = get_fuccisa_default_sensor()
thresholds = [0.1, 0.1]

def get_fucciphase_class(labels, image):
    props = regionprops_table(labels, image,
                              properties=['label', 'mean_intensity'])
    data = pd.DataFrame(props)
    background_label = invert(labels > 0)
    background_label = background_label.astype(np.uint8)
    bg_props = regionprops_table(background_label, image,
                                 properties=['mean_intensity'])
    bg_props_df = pd.DataFrame(bg_props)
    # has only one entry per channel because background is ignored
    background = bg_props_df.iloc[0].to_numpy()
    estimate_cell_phase_from_max_intensity(
        data,
        channels,
        sensor,
        background=background[:2],  # only first two channels are bg
        thresholds=thresholds,
    )

    data["phase"] = data["DISCRETE_PHASE_MAX"].map(convert_categories)
    cls_dict = {}
    for _, row in data[["label", "phase"]].iterrows():
        cls_dict[int(row["label"])] = int(row["phase"])
    return cls_dict

for x in tqdm(X):
    Y_val_pred, res = model_1d.predict_instances(x[..., 2])
    labels_1d.append(Y_val_pred)
    classes_1d.append(class_from_res(res))
    
    Y_val_pred, res = model_2d.predict_instances(x[..., 0:2])
    labels_2d.append(Y_val_pred)
    classes_2d.append(class_from_res(res))
    classes_fucciphase.append(get_fucciphase_class(Y_val_pred, x))

    Y_val_pred, res = model_3d.predict_instances(x)
    labels_3d.append(Y_val_pred)
    classes_3d.append(class_from_res(res))


taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats_1d = [
    matching_dataset(Y, labels_1d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

stats_2d = [
    matching_dataset(Y, labels_2d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

stats_3d = [
    matching_dataset(Y, labels_3d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

# get segmentation accuracy
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

    plt.savefig(f"classifiction_nw_{m}.pdf")
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

plt.savefig("classifiction_nw_label_numbers.pdf")
plt.show()

print("Stats at 0.5 IoU: ", stats_1d[4])
print("Stats at 0.5 IoU: ", stats_2d[4])
print("Stats at 0.5 IoU: ", stats_3d[4])

results_dict = {}
for i in range(3):
    results_dict[f"class_gt_{i + 1}"] = 0
    results_dict[f"class_pred_{i + 1}"] = 0
# check classification
for y_1ch, y_2ch, y_3ch, y_gt, c_1ch, c_2ch, c_3ch, c_fp, c_gt in zip(labels_1d, labels_2d, labels_3d, Y, classes_1d, classes_2d, classes_3d, classes_fucciphase, C):
    with open(c_gt) as fp:
        cls_gt = json.load(fp)
    # get labels overlapping enough with GT
    _, matching_labels = group_matching_labels([y_gt, y_2ch], thresh=0.5)
    label_pairs = np.column_stack([matching_labels.flatten(), y_2ch.flatten()])
    print(label_pairs.shape)
    label_pairs = np.unique(label_pairs, axis=0)
    print(label_pairs.shape)
    print(label_pairs)
    all_gt_labels = np.unique(y_gt)
    matching_labels = np.unique(matching_labels)
    print(all_gt_labels)
    print(matching_labels)
    common_labels = np.intersect1d(all_gt_labels, matching_labels)
    print(common_labels)
    for label in common_labels:
        if label == 0:
            continue
        pred_label = label_pairs[label_pairs[:, 0] == label][0][1]
        print(label)
        print(pred_label)
        try:
            predicted_class = c_fp[pred_label]
        except KeyError:
            predicted_class = c_fp[str(pred_label)]
        gt_class = cls_gt[str(label)]
        results_dict[f"class_gt_{gt_class}"] += 1
        results_dict[f"class_pred_{predicted_class}"] += 1
print(results_dict)
