from aicsimageio import AICSImage
import json
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops_table
import napari
import yaml

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)

# Channels
channels = metadata["channels"]
filename = metadata["filename"]
y_crop = [250, 750]
x_crop = [400, 900]

img_stream = AICSImage(filename)

img_tub = img_stream.get_image_dask_data("YX", S=int(channels["tubulin"]), T=-1)
img_cyan = img_stream.get_image_dask_data("YX", S=int(channels["cyan"]), T=-1)
img_magenta = img_stream.get_image_dask_data("YX", S=int(channels["magenta"]), T=-1)

label_file = "gt_last_frame_relabeled.tif"

viewer = napari.Viewer()

def phase_from_class(class_id):
    if class_id == 1:
        return "G1"
    elif class_id == 2:
        return "S"
    elif class_id == 3:
        return "G2/M"
    else:
        return "N/A"
        raise ValueError(f"No valid class for ID {class_id}")

def phases_from_class_list(class_id_list):
    return [phase_from_class(class_id) for class_id in class_id_list]


textkwargs={"size": 16}

labels = imread(label_file)
with open(label_file.replace(".tif", ".json")) as fp:
    classes = json.load(fp)
props = regionprops_table(labels, properties=("label", "centroid"))
data = pd.DataFrame(props)
class_ids = []
points = []
for _, row in data.iterrows():
    label = str(round(row["label"]))
    centroid_0 = row["centroid-0"]
    centroid_1 = row["centroid-1"]
    print(centroid_0)
    if not y_crop[0] < centroid_0 < y_crop[1]:
        continue
    if not x_crop[0] < centroid_1 < x_crop[1]:
        continue
    centroid = [centroid_0 - y_crop[0], centroid_1 - x_crop[0]]
    class_ids.append(int(classes[label]))
    points.append(centroid)
                 
cyan_layer = viewer.add_image(img_cyan[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]], blending="additive", colormap="cyan", name="PIP")
magenta_layer = viewer.add_image(img_magenta[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]], blending="additive", colormap="magenta", name="Geminin")
tub_layer=viewer.add_image(img_tub[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]], colormap="green", blending="additive", name="PCNA")
labels_layer = viewer.add_labels(labels[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]])
labels_layer.contour = 2


class_name = phases_from_class_list(class_ids)

pts = viewer.add_points(
        points,
        features={"class": class_name},
        text={"string": "{class}",
              "color": "red",
              **textkwargs},
        size=0.01,
    )
napari.run()
