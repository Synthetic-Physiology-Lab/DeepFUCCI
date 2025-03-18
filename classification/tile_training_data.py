from skimage.io import imsave, imread
import numpy as np
from glob import glob
from csbdeep.utils import Path
import os
import json

# data_dir = "training_data"
data_dir = "training_data_relabeled_classified"  
new_data_dir = "training_data_tiled_classified"

patch_size = 256
minimum_required_masks = 3

if not os.path.isdir(new_data_dir):
    os.mkdir(new_data_dir)
if not os.path.isdir(os.path.join(new_data_dir, "images")):
    os.mkdir(os.path.join(new_data_dir, "images"))
if not os.path.isdir(os.path.join(new_data_dir, "masks")):
    os.mkdir(os.path.join(new_data_dir, "masks"))
if not os.path.isdir(os.path.join(new_data_dir, "classes")):
    os.mkdir(os.path.join(new_data_dir, "classes"))

classes_directory = "classes_manually_fixed"
image_files = sorted(glob(os.path.join(data_dir, "images", "*.tif")))
mask_files = sorted(glob(os.path.join(data_dir, "masks", "*.tif")))
classes_files = sorted(glob(os.path.join(data_dir, classes_directory, "*.json")))
assert len(image_files) == len(mask_files)
assert len(mask_files) == len(classes_files)
assert all(Path(x).name == Path(y).name for x, y in zip(image_files, mask_files))
assert all(Path(x).name.replace(".json", "") == Path(y).name.replace(".tif", "") for x, y in zip(classes_files, mask_files))

number_of_masks = 0

for image_file, mask_file, class_file in zip(image_files, mask_files, classes_files):
    nucleus_labels = imread(mask_file)
    assert np.max(nucleus_labels) < 65500
    nucleus_labels = nucleus_labels.astype(np.uint16)
    image = imread(image_file)
    with open(class_file) as fp:
        original_cls_dict = json.load(fp)
    # Assert quadratic shape
    assert nucleus_labels.shape[0] == nucleus_labels.shape[1]
    x_shape = nucleus_labels.shape[0] 
    tiles_per_row = x_shape // patch_size
    tiles = tiles_per_row ** 2
    print(tiles_per_row, tiles)
    if tiles == 1:
        # here we assume that there are always enough masks
        number_of_masks += len(np.unique(nucleus_labels)) - 1
        imsave(os.path.join(new_data_dir, "masks", Path(mask_file).name), nucleus_labels, dtype=np.uint16, compression="zlib", check_contrast=False)
        imsave(os.path.join(new_data_dir, "images", Path(image_file).name), image, compression="zlib")
        # just copy the class labels
        basename = os.path.basename(image_file)
        basename = basename.replace(".tif", ".json")
        unique_masks = np.unique(nucleus_labels)
        cls_dict = {}
        for mask_label in unique_masks:
            if mask_label == 0:
                continue
            try:
                # item to get native python type
                cls_dict[mask_label.item()] = int(original_cls_dict[str(mask_label)])
            except KeyError:
                print(f"Problem in {class_file}, label {mask_label} not found.")
                raise
        with open(os.path.join(new_data_dir, "classes", Path(class_file).name), "w") as fp:
            json.dump(cls_dict, fp)
        continue

    x_width = x_shape // tiles_per_row
    y_width = x_width
    
    tile = 1 
    for i in range(tiles_per_row):
        for j in range(tiles_per_row):
            x_low = i * x_width
            x_high = (i + 1) * x_width
            y_low = j * y_width
            y_high = (j + 1) * y_width
            assert x_high - x_low >= patch_size
            new_masks = nucleus_labels[y_low:y_high, x_low:x_high]
            assert new_masks.shape[0] >= patch_size
            # do not write empty parts
            if np.all(np.isclose(new_masks, 0)):
                tile += 1
                continue
            mask_file_name = Path(mask_file).name
            mask_file_name = mask_file_name.replace(".tif", f"_{tile}.tif")
            image_file_name = Path(image_file).name
            image_file_name = image_file_name.replace(".tif", f"_{tile}.tif")
            class_file_name = Path(class_file).name
            class_file_name = class_file_name.replace(".json", f"_{tile}.json")
            # do not save data if not enough pixels
            unique_masks = np.unique(new_masks)
            if len(unique_masks) < minimum_required_masks + 1:
                print(f"Not enough masks in tile [{i}, {j}] in file {image_file_name}")
                continue

            number_of_masks += len(np.unique(new_masks)) - 1
            imsave(os.path.join(new_data_dir, "masks", mask_file_name), new_masks, dtype=np.uint16, compression="zlib", check_contrast=False)
            imsave(os.path.join(new_data_dir, "images", image_file_name), image[y_low:y_high, x_low:x_high, :], compression="zlib")
            cls_dict = {}
            for mask_label in unique_masks:
                if mask_label == 0:
                    continue
                try:
                    # item to get native python type
                    cls_dict[mask_label.item()] = int(original_cls_dict[str(mask_label)])
                except KeyError:
                    print(f"Problem in {class_file}, label {mask_label} not found.")
                    raise
            with open(os.path.join(new_data_dir, "classes", class_file_name), "w") as fp:
                json.dump(cls_dict, fp)
            tile += 1

print(f"The training dataset contains {number_of_masks} labels")
