import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from csbdeep.utils import normalize
from bioio import BioImage
import bioio_ome_tiff
import bioio_tifffile
from pathlib import Path

from stardist import (
    fill_label_holes,
)
from stardist.matching import matching_dataset
from instanseg import InstanSeg

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

filename = metadata["filename"]
if not Path(filename).exists():
    filename = Path("../../../data/HaCaT_Han_et_al") / filename


img_stream = BioImage(filename, reader=bioio_ome_tiff.Reader)
label_stream = BioImage("labels_manual_annotation.tif", reader=bioio_tifffile.Reader)

X = []
Y = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_1d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_2d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)
labels_3d = np.zeros(
    shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16
)


pixel_size = img_stream.physical_pixel_sizes.X
# rescale from um to mm
pixel_size *= 1e-3
print(pixel_size)
instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1)


for T in tqdm(range(img_stream.dims.T)):
    img_tubulin = img_stream.get_image_data("YX", C=int(channels["tubulin"]), T=T)
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)
    gt_labels = label_stream.get_image_data("YX", Z=T)
    Y[T] = gt_labels[:]

    # normalize image
    img_tubulin = normalize(img_tubulin, pmin=1)
    img_cyan = normalize(img_cyan, pmin=1)
    img_magenta = normalize(img_magenta, pmin=1)

    X.append(img_tubulin)
    # 1 channel model
    labels = instanseg_fluorescence.eval_small_image(
        image=img_tubulin,
        pixel_size=pixel_size,
        return_image_tensor=False,
        target="nuclei",
    )
    labels_1d[T] = labels[:]

    # 2 channel model
    labels = instanseg_fluorescence.eval_small_image(
        image=np.stack([img_cyan, img_magenta]),
        pixel_size=pixel_size,
        return_image_tensor=False,
        target="nuclei",
    )
    labels_2d[T] = labels[:]

    # 3 channel model
    labels = instanseg_fluorescence.eval_small_image(
        image=np.stack([img_cyan, img_magenta, img_tubulin]),
        pixel_size=pixel_size,
        return_image_tensor=False,
        target="nuclei",
    )
    labels_3d[T] = labels[:]

Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
print("number of images: %3d" % len(X))

i = 0
for Y_val_pred in [labels_1d, labels_2d, labels_3d]:
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [
        matching_dataset(Y, Y_val_pred, thresh=t, show_progress=False)
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
    plt.show()

    print("Stats at 0.5 IoU for {i} CH: ", stats[4])
    i += 1
