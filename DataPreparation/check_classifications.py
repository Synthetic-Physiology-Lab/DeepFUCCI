import json

import pandas as pd
import glob

from skimage.io import imread
from skimage.measure import regionprops_table
import napari

image_files = glob.glob("images/*.tif")

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


textkwargs={"size": 16}

# go through all files
for idx, image_file in enumerate(image_files):
    print(f"File {idx}, {image_file}")
    image = imread(image_file)
    labels = imread(image_file.replace("images", "masks"))
    with open(image_file.replace("images", "classes").replace(".tif", ".json")) as fp:
        classes = json.load(fp)
    props = regionprops_table(labels, properties=("label", "centroid"))
    data = pd.DataFrame(props)
    class_ids = []
    points = []
    for _, row in data.iterrows():
        label = str(round(row["label"]))
        centroid = [row["centroid-0"], row["centroid-1"]]
        class_ids.append(classes[label])
        points.append(centroid)
                     
    if idx == 0:
        cyan_layer = viewer.add_image(image[..., 0], blending="additive", colormap="cyan")
        magenta_layer = viewer.add_image(image[..., 1], blending="additive", colormap="magenta")
    else:
        cyan_layer.data = image[..., 0]
        magenta_layer.data = image[..., 1]


    if idx == 0:
        labels_layer = viewer.add_labels(labels)
    else:
        labels_layer.data = labels
    labels_layer.contour = 2


    class_name = phases_from_class_list(class_ids)

    pts = viewer.add_points(
            points,
            features={"class": class_name},
            text={"string": "{class}",
                  "color": "white",
                  **textkwargs},
            size=0.01,
        )
    # hit input to open next file
    input()
napari.run()
