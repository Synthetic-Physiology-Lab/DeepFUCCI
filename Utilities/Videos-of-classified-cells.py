import json
import napari
from bioio import BioImage
import bioio_nd2
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops_table
import pandas as pd
import yaml
import os
from csbdeep.utils import normalize
from napari_animation import Animation

viewer = napari.Viewer()


def phase_from_class(class_id):
    if class_id == 0:
        return "BG"
    if class_id == 1:
        return "G1"
    elif class_id == 2:
        return "G1/S"
    elif class_id == 3:
        return "S/G2/M"
    else:
        raise ValueError(f"No valid class: {class_id}")


def phases_from_class_list(class_id_list):
    return [phase_from_class(class_id) for class_id in class_id_list]


def prepare_bboxes_for_napari(masks, classes):
    """Assumption: masks have shape TYX"""
    if not masks.shape[0] == len(classes):
        raise ValueError(
            "Provide masks in format TYX" " and classes in list with length T"
        )

    points = []
    new_labels = np.zeros(masks.shape, dtype=np.uint16)
    class_names = []
    all_bboxes = []
    all_bbox_colors = []
    for t in range(masks.shape[0]):
        class_ids = []
        props = regionprops_table(masks[t], properties=("label", "centroid", "bbox"))
        data = pd.DataFrame(props)
        for _, row in data.iterrows():
            label = round(row["label"])
            centroid = [t, row["centroid-0"], row["centroid-1"]]
            try:
                class_id = classes[t][label]
            except KeyError:
                class_id = classes[t][str(label)]
            bbox = [row[f"bbox-{i}"] for i in range(4)]
            # bbox: 0 - miny, 1 - minx, 2 - maxy, 3- maxx
            bbox_rect = [
                [t, bbox[0], bbox[1]],
                [t, bbox[2], bbox[1]],
                [t, bbox[2], bbox[3]],
                [t, bbox[0], bbox[3]],
            ]
            all_bboxes.append(bbox_rect)
            if class_id == 1:
                all_bbox_colors.append("cyan")
                new_labels[t, masks[t] == label] = 2
            elif class_id == 2:
                all_bbox_colors.append("brown")
                new_labels[t, masks[t] == label] = 1
            elif class_id == 3:
                all_bbox_colors.append("magenta")
                new_labels[t, masks[t] == label] = 3
            else:
                # background case
                all_bbox_colors.append("black")
            class_ids.append(class_id)
            points.append(centroid)

        class_names.extend(phases_from_class_list(class_ids))

    return (new_labels, points, class_names, all_bboxes, all_bbox_colors)


def read_json_file(json_file):
    with open(json_file) as fp:
        data = json.load(fp)
    return data


def add_bbox_layers_to_napari(
    new_labels, points, class_name, bbox_rects, bbox_rect_colors
):
    viewer.add_labels(new_labels, name="new")
    viewer.layers[-1].visible = False
    textkwargs = {"size": 16}

    pts = viewer.add_points(
        points,
        features={"class": class_name},
        text={"string": "{class}", "color": "white", **textkwargs},
        size=0.01,
    )
    viewer.layers[-1].visible = False

    shapes_layer = viewer.add_shapes(
        bbox_rects,
        face_color="transparent",
        edge_color=bbox_rect_colors,
        name="bounding boxes",
        edge_width=12,
    )
    return pts, shapes_layer


with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]

flatfield_correction = True
for channel in channels:
    if not os.path.exists(f"basic_model_{channel}"):
        flatfield_correction = False
        print("Flatfield correction will not be applied")


filename = metadata["filename"]

img_stream = BioImage(filename, reader=bioio_nd2.Reader)

X = img_stream.get_image_dask_data(
    "TYXC",
    C=[int(channels["cyan"]), int(channels["magenta"]), int(channels["tubulin"])],
).compute()
X = normalize(X, axis=(1, 2), pmin=1, pmax=99.8)
Y_1CH = imread("stardist_labels_1_channel_classifier.tif")
Y_2CH = imread("stardist_labels_2_channel_classifier.tif")
Y_3CH = imread("stardist_labels_3_channel_classifier.tif")
classes_1CH = read_json_file("classes_1_channel.json")
classes_2CH = read_json_file("classes_2_channel.json")
classes_3CH = read_json_file("classes_3_channel.json")

print("number of images: %3d" % len(X))

cyan_layer = viewer.add_image(
    X[..., 0], colormap="cyan", blending="additive", name="cyan"
)
magenta_layer = viewer.add_image(
    X[..., 1], colormap="magenta", blending="additive", name="magenta"
)
tubulin_layer = viewer.add_image(X[..., 2], blending="additive", name="tubulin")
viewer.layers[-1].visible = False

lab_1ch_layer = viewer.add_labels(Y_1CH, name="1CH")
viewer.layers[-1].visible = False

lab_2ch_layer = viewer.add_labels(Y_2CH, name="2CH")
viewer.layers[-1].visible = False

lab_3ch_layer = viewer.add_labels(Y_3CH, name="3CH")
viewer.layers[-1].visible = False


(
    new_labels_1_CH,
    points_1_CH,
    class_names_1_CH,
    all_bbox_rects_1_CH,
    all_bbox_colors_1_CH,
) = prepare_bboxes_for_napari(Y_1CH, classes_1CH)

pts_1CH, shapes_1CH = add_bbox_layers_to_napari(
    new_labels_1_CH,
    points_1_CH,
    class_names_1_CH,
    all_bbox_rects_1_CH,
    all_bbox_colors_1_CH,
)

(
    new_labels_2_CH,
    points_2_CH,
    class_names_2_CH,
    all_bbox_rects_2_CH,
    all_bbox_colors_2_CH,
) = prepare_bboxes_for_napari(Y_2CH, classes_2CH)

pts_2CH, shapes_2CH = add_bbox_layers_to_napari(
    new_labels_2_CH,
    points_2_CH,
    class_names_2_CH,
    all_bbox_rects_2_CH,
    all_bbox_colors_2_CH,
)

(
    new_labels_3_CH,
    points_3_CH,
    class_names_3_CH,
    all_bbox_rects_3_CH,
    all_bbox_colors_3_CH,
) = prepare_bboxes_for_napari(Y_3CH, classes_3CH)

pts_3CH, shapes_3CH = add_bbox_layers_to_napari(
    new_labels_3_CH,
    points_3_CH,
    class_names_3_CH,
    all_bbox_rects_3_CH,
    all_bbox_colors_3_CH,
)

animation = Animation(viewer)

tubulin_layer.visible = True
cyan_layer.visible = False
magenta_layer.visible = False

lab_1ch_layer.visible = False
lab_2ch_layer.visible = False
lab_3ch_layer.visible = False

shapes_1CH.visible = True
shapes_2CH.visible = False
shapes_3CH.visible = False

# start animation on first frame
viewer.dims.current_step = (0, 0, 0)
animation.capture_keyframe()
# last frame
viewer.dims.current_step = (Y_1CH.shape[0], 0, 0)
animation.capture_keyframe(steps=Y_1CH.shape[0] - 1)
animation.animate("movie_1ch.mp4", canvas_only=True, fps=4, quality=9, scale_factor=1.0)

animation = Animation(viewer)

tubulin_layer.visible = False
cyan_layer.visible = True
magenta_layer.visible = True

lab_1ch_layer.visible = False
lab_2ch_layer.visible = False
lab_3ch_layer.visible = False

shapes_1CH.visible = False
shapes_2CH.visible = True
shapes_3CH.visible = False

# start animation on first frame
viewer.dims.current_step = (0, 0, 0)
animation.capture_keyframe()
# last frame
viewer.dims.current_step = (Y_1CH.shape[0], 0, 0)
animation.capture_keyframe(steps=Y_1CH.shape[0] - 1)
animation.animate("movie_1ch.mp4", canvas_only=True, fps=4, quality=9, scale_factor=1.0)


animation = Animation(viewer)

tubulin_layer.visible = True
cyan_layer.visible = False
magenta_layer.visible = False

lab_1ch_layer.visible = False
lab_2ch_layer.visible = False
lab_3ch_layer.visible = False

shapes_1CH.visible = True
shapes_2CH.visible = False
shapes_3CH.visible = False

# start animation on first frame
viewer.dims.current_step = (0, 0, 0)
animation.capture_keyframe()
# last frame
viewer.dims.current_step = (Y_1CH.shape[0], 0, 0)
animation.capture_keyframe(steps=Y_1CH.shape[0] - 1)
animation.animate("movie_2ch.mp4", canvas_only=True, fps=4, quality=9, scale_factor=1.0)

animation = Animation(viewer)

tubulin_layer.visible = True
cyan_layer.visible = True
magenta_layer.visible = True

lab_1ch_layer.visible = False
lab_2ch_layer.visible = False
lab_3ch_layer.visible = False

shapes_1CH.visible = False
shapes_2CH.visible = False
shapes_3CH.visible = True

# start animation on first frame
viewer.dims.current_step = (0, 0, 0)
animation.capture_keyframe()
# last frame
viewer.dims.current_step = (Y_1CH.shape[0], 0, 0)
animation.capture_keyframe(steps=Y_1CH.shape[0] - 1)
animation.animate("movie_2ch.mp4", canvas_only=True, fps=4, quality=9, scale_factor=1.0)


napari.run()
