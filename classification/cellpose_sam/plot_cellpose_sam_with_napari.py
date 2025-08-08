import napari
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.io import imread
from csbdeep.utils import Path, normalize
from skimage.measure import regionprops_table
import pandas as pd
from scipy.stats import mode

viewer = napari.Viewer()


def phase_from_class(class_id):
    if class_id == 1:
        return "G1"
    elif class_id == 2:
        return "G1/S"
    elif class_id == 3:
        return "S/G2/M"
    else:
        raise ValueError("No valid class")


def phases_from_class_list(class_id_list):
    return [phase_from_class(class_id) for class_id in class_id_list]


def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[..., 0]
    minc = bbox_extents[..., 1]
    maxr = bbox_extents[..., 2]
    maxc = bbox_extents[..., 3]

    bbox_rect = np.array([[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]])
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect


X = sorted(glob("images/*.tif"))
Y = sorted(glob("masks/*.tif"))
assert len(X) > 0
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))

axis_norm = (0, 1)  # normalize channels independently
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]

# crop
x_crop = slice(125, 450)
y_crop = slice(0, 325)

X = [x[y_crop, x_crop, :] for x in X]
Y = [y[y_crop, x_crop] for y in Y]

Y_val_pred = [imread("masks_cp_sam.tif")]
class_masks = imread("classes_cp_sam.tif")

Y_val_pred = [Y_val_pred[0][y_crop, x_crop]]
class_masks = class_masks[y_crop, x_crop]

img = X[0]
lbl = Y[0]

viewer.add_image(img[..., 0], colormap="cyan", blending="additive", name="cyan")
viewer.layers[-1].contrast_limits = [0.11794248894837342, 1.0]
viewer.add_image(img[..., 1], colormap="magenta", blending="additive", name="magenta")
viewer.layers[-1].contrast_limits = [0.23591429122576585, 1.1486881971359253]
viewer.add_image(img[..., 2], blending="additive", name="tubulin")
viewer.layers[-1].contrast_limits = [0.18208224419504404, 1.6865873336791992]
viewer.export_figure("all_3_channels_no_bbox.png")
viewer.layers[-1].visible = False

m_n = normalize(img[..., 0])
c_n = normalize(img[..., 1])
dapieq = np.maximum(m_n, c_n)

viewer.add_labels(lbl, name="GT")
viewer.layers[-1].visible = False
viewer.add_labels(Y_val_pred[0], name="Pred")
viewer.layers[-1].visible = False
viewer.add_image(dapieq, colormap="gray", blending="additive", name="dapieq")
viewer.layers[-1].visible = False


def class_from_res(class_masks, masks):
    cls_dict = {}
    all_labels = np.unique(masks)
    print(all_labels)
    for mask in all_labels:
        if mask == 0:
            continue
        class_id = mode(class_masks[masks == mask])[0]
        cls_dict[int(mask)] = int(class_id)
    return cls_dict


props = regionprops_table(Y_val_pred[0], properties=("label", "centroid", "bbox"))
data = pd.DataFrame(props)
class_ids = []
points = []
classes = class_from_res(class_masks, Y_val_pred[0])
print(classes)
new_labels = np.zeros(Y_val_pred[0].shape, dtype=np.uint16)
bbox_rects_g1 = []
bbox_rects_g1s = []
bbox_rects_sg2m = []
for _, row in data.iterrows():
    label = round(row["label"])
    centroid = [row["centroid-0"], row["centroid-1"]]
    class_id = classes[label]
    class_ids.append(class_id)
    if class_id == 1:
        bbox_rects_g1.append([row[f"bbox-{i}"] for i in range(4)])
        new_labels[Y_val_pred[0] == label] = 2
    elif class_id == 2:
        bbox_rects_g1s.append([row[f"bbox-{i}"] for i in range(4)])
        new_labels[Y_val_pred[0] == label] = 1
    elif class_id == 3:
        bbox_rects_sg2m.append([row[f"bbox-{i}"] for i in range(4)])
        new_labels[Y_val_pred[0] == label] = 3
    points.append(centroid)

print("Preparing visualization")
bbox_rects_g1 = np.array(bbox_rects_g1)
bbox_rects_g1 = make_bbox(bbox_rects_g1)
bbox_rects_g1s = np.array(bbox_rects_g1s)
bbox_rects_g1s = make_bbox(bbox_rects_g1s)
bbox_rects_sg2m = np.array(bbox_rects_sg2m)
bbox_rects_sg2m = make_bbox(bbox_rects_sg2m)

viewer.add_labels(new_labels, name="new")
viewer.layers[-1].visible = False
class_name = phases_from_class_list(class_ids)
textkwargs = {"size": 16}

pts = viewer.add_points(
    points,
    features={"class": class_name},
    text={"string": "{class}", "color": "white", **textkwargs},
    size=0.01,
)
viewer.layers[-1].visible = False

shapes_layer_g1 = viewer.add_shapes(
    bbox_rects_g1,
    face_color="transparent",
    edge_color="cyan",
    name="bounding box g1",
    edge_width=3,
)

shapes_layer_g1s = viewer.add_shapes(
    bbox_rects_g1s,
    face_color="transparent",
    edge_color="brown",
    name="bounding box g1s",
    edge_width=3,
)

shapes_layer_sg2m = viewer.add_shapes(
    bbox_rects_sg2m,
    face_color="transparent",
    edge_color="magenta",
    name="bounding box sg2m",
    edge_width=3,
)
viewer.export_figure("bboxes_3_channels_cpsam.png")
napari.run()
