import numpy as np
import matplotlib
from tqdm import tqdm
from skimage.io import imread
from skimage.measure import label as label_skimage
from csbdeep.utils import normalize
from pathlib import Path

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

data_file = "image_cyan_magenta_last_frame.tif"
if not Path(data_file).exists():
    data_file = Path("../../../../data/test_cellmaptracer") / data_file
    if not Path(data_file).exists():
        raise FileNotFoundError("Data file not there")

X = [np.moveaxis(imread(data_file), 0, -1)]
Y = [imread("gt_last_frame.tif")]

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

model_2d = StarDist2D(
    None, name="stardist_2_channel_latest", basedir=Path.home() / "models"
)

scale = 0.65 / 0.34
print("Scaling with factor: ", scale)

Y_val_pred_2d = [model_2d.predict_instances(x, scale=scale)[0] for x in tqdm(X)]

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats_2d = [
    matching_dataset(Y, Y_val_pred_2d, thresh=t, show_progress=False)
    for t in tqdm(taus)
]

print("Stats at 0.5 IoU for 2 CH :", stats_2d[4])
