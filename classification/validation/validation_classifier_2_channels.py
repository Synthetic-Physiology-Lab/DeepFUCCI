import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from csbdeep.utils import normalize

from stardist import (
    fill_label_holes,
    gputools_available,
)
from stardist.matching import matching_dataset
from stardist.models import StarDist2D

training_data_dir = "training_data_tiled_strict_classified_new"
# use the same data split as in training
with open(f"{training_data_dir}/dataset_split.json") as fp:
    dataset_split = json.load(fp)

def read_json(path):
    with open(path) as fp:
        data = json.load(fp)
    data_list = []
    for key, value in sorted(data.items()):
        data_list.append((int(key), value))
        
    return dict(data_list)

X = [imread(f"{training_data_dir}/images/{img_name}") for img_name in dataset_split["validation"]]
Y = [fill_label_holes(imread(f"{training_data_dir}/masks/{img_name}")) for img_name in dataset_split["validation"]]
C_val = [read_json(f"{training_data_dir}/classes/{img_name.replace('.tif', '.json')}") for img_name in dataset_split["validation"]]


n_channel = 2
n_classes = 3

axis_norm = (0, 1)  # normalize channels independently
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

X_val = [normalize(x[..., 0:2], 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y_val = [fill_label_holes(y) for y in tqdm(Y)]

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = gputools_available()
print("Using GPU: ", use_gpu)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.1, total_memory=50000)

model = StarDist2D(None, name="stardist_multiclass", basedir="training_2_channels_stardist_classifier/models")

# the class object ids are stored in the 'results' dict and correspond to the label ids in increasing order 
def class_from_res(res):
    cls_dict = dict((i+1,c) for i,c in enumerate(res['class_id']))
    return cls_dict

Y_val_pred, res_val_pred = tuple(zip(*[model.predict_instances(x)
              for x in tqdm(X_val[:])]))

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

precision_at_0_5_IOU = []
accuracy_at_0_5_IOU = []
tp_at_0_5_IOU = []
fp_at_0_5_IOU = []
fn_at_0_5_IOU = []
class_id_pairs = []
for class_id in range(1, 4):
    for class_id_pred in range(1, 4):
        class_id_pairs.append((class_id, class_id_pred))
        class_y_vals = []
        class_y_vals_pred = []
        for y_val, class_ids, y_val_pred, class_ids_pred in zip(Y_val, C_val, Y_val_pred, res_val_pred):
            class_ids_pred = class_from_res(class_ids_pred)
            new_y_val = np.zeros(y_val.shape, y_val.dtype)
            new_y_val_pred = np.zeros(y_val_pred.shape, y_val_pred.dtype)
            new_y_val[:] = y_val[:]
            new_y_val_pred[:] = y_val_pred[:]
            for label in class_ids:
                if class_ids[label] != class_id:
                    new_y_val[new_y_val == label] = 0
            for label in class_ids_pred:
                if class_ids_pred[label] != class_id_pred:
                    new_y_val_pred[new_y_val_pred == label] = 0
            class_y_vals.append(new_y_val)
            class_y_vals_pred.append(new_y_val_pred)

        stats = [
            matching_dataset(class_y_vals, class_y_vals_pred, thresh=t, show_progress=False)
            for t in tqdm(taus)
        ]
        precision_at_0_5_IOU.append(stats[4]._asdict()["precision"])
        accuracy_at_0_5_IOU.append(stats[4]._asdict()["accuracy"])
        fp_at_0_5_IOU.append(stats[4]._asdict()["fp"])
        fn_at_0_5_IOU.append(stats[4]._asdict()["fn"])
        tp_at_0_5_IOU.append(stats[4]._asdict()["tp"])

        if class_id == class_id_pred:
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
            plt.savefig(f"metrics_2ch_class_{class_id}.png")
            plt.close()

match_iou_0_5 = matching_dataset(Y, Y_val_pred, thresh=0.5, show_progress=False)
print("Pairs:")
print(class_id_pairs)
print("Precision:")
print(match_iou_0_5._asdict()["precision"])
print(precision_at_0_5_IOU)
print("Accuracy:")
print(match_iou_0_5._asdict()["accuracy"])
print(accuracy_at_0_5_IOU)
print("TP:")
print(match_iou_0_5._asdict()["tp"])
print(sum(tp_at_0_5_IOU))
print(tp_at_0_5_IOU)
print("FP:")
print(match_iou_0_5._asdict()["fp"])
print(fp_at_0_5_IOU)
print("FN:")
print(match_iou_0_5._asdict()["fn"])
print(fn_at_0_5_IOU)
