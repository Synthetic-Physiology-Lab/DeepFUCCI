import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from scipy.stats import mode
import torch
from torch import nn
from pathlib import Path

from stardist import (
    fill_label_holes,
    gputools_available,
)
from stardist.matching import matching_dataset
from cellpose import vit_sam, models

# --- Start of Cellpose-specific initialization ---

def initialize_class_net(nclasses=3, device=torch.device("cuda")):
    """Initializes the classification network for Cellpose."""
    net = vit_sam.Transformer(rdrop=0.4).to(device)
    # default model
    net.load_model(Path.home() / ".cellpose/models/cpsam", device=device, strict=False)

    # initialize weights for class maps
    ps = 8  # patch size
    nout = 3
    w0 = net.out.weight.data.detach().clone()
    b0 = net.out.bias.data.detach().clone()
    net.out = nn.Conv2d(256, (nout + nclasses + 1) * ps**2, kernel_size=1).to(device)
    # set weights for background map
    i = 0
    net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = (
        -0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
    )
    net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[
        (nout - 1) * ps**2 : nout * ps**2
    ]
    # set weights for maps to nuclei classes
    for i in range(1, nclasses + 1):
        net.out.weight.data[i * ps**2 : (i + 1) * ps**2] = (
            0.5 * w0[(nout - 1) * ps**2 : nout * ps**2]
        )
        net.out.bias.data[i * ps**2 : (i + 1) * ps**2] = b0[
            (nout - 1) * ps**2 : nout * ps**2
        ]
    net.out.weight.data[-(nout * ps**2) :] = w0
    net.out.bias.data[-(nout * ps**2) :] = b0
    net.W2 = nn.Parameter(
        torch.eye((nout + nclasses + 1) * ps**2).reshape(
            (nout + nclasses + 1) * ps**2, nout + nclasses + 1, ps, ps
        ),
        requires_grad=False,
    )
    net.to(device)
    return net

# --- End of Cellpose-specific initialization ---


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


n_classes = 3

X_val = [x[..., 0:2] for x in tqdm(X)]
Y_val = [fill_label_holes(y) for y in tqdm(Y)]

# Use GPU if available
use_gpu = torch.cuda.is_available()
print("Using GPU: ", use_gpu)

# --- Replace StarDist model with Cellpose model ---
device = torch.device("cuda" if use_gpu else "cpu")
model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
net = initialize_class_net(nclasses=n_classes, device=device)
# Make sure to provide the correct path to your trained Cellpose model weights
net.load_model(
    "models/FUCCI_batch_size_8",
    device=device,
    strict=False,
)
net.eval()
model.net = net
model.net_ortho = None


# --- Define conversion function and predict with Cellpose ---

def class_from_res_cellpose(class_masks, masks):
    """
    Converts Cellpose's class mask and label mask to a StarDist-style
    dictionary mapping label IDs to class IDs.
    """
    cls_dict = {}
    all_labels = np.unique(masks)
    for label in all_labels:
        if label == 0:
            continue
        # Find the most frequent class in the area of the label
        class_id = mode(class_masks[masks == label], keepdims=False)[0]
        cls_dict[int(label)] = int(class_id)
    return cls_dict

# Perform prediction with Cellpose for each validation image
Y_val_pred = []
res_val_pred = []
for x in tqdm(X_val[:]):
    masks, flows, styles = model.eval(
        [x],
        diameter=None,
        augment=False,
        bsize=256,
        tile_overlap=0.1,
        batch_size=64,
        flow_threshold=0.4,
        cellprob_threshold=0,
    )
    classes_pred = [s.squeeze().argmax(axis=-1) for s in styles]
    
    # Assuming single image prediction, so we take the first element
    y_pred = masks[0]
    class_mask_pred = classes_pred[0]
    
    Y_val_pred.append(y_pred)
    
    # Convert to StarDist dictionary format
    class_dict = class_from_res_cellpose(class_mask_pred, y_pred)
    res_val_pred.append(class_dict)


taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

accuracy_at_0_5_IOU = []
class_id_pairs = []
for class_id in range(1, 4):
    for class_id_pred in range(1, 4):
        class_id_pairs.append((class_id, class_id_pred))
        class_y_vals = []
        class_y_vals_pred = []
        # The structure of this loop remains unchanged
        for y_val, class_ids, y_val_pred, class_ids_pred in zip(Y_val, C_val, Y_val_pred, res_val_pred):
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
        accuracy_at_0_5_IOU.append(stats[4]._asdict()["accuracy"])

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
            plt.savefig(f"metrics_cellpose_sam_1ch_class_{class_id}.png")
            plt.close()

print("Pairs:")
print(class_id_pairs)
print("Accuracy:")
print(accuracy_at_0_5_IOU)
