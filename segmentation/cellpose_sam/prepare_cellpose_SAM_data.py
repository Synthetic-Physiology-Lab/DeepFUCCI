from cellpose import transforms, dynamics
import torch
from glob import glob
import os
from cellpose import io
import json
import tifffile
import fastremap
from pathlib import Path

logger, log_file = io.logger_setup()


def convert_fucci_data(data_dir, out_dir, n_classes=3):
    img_files = [f for f in sorted(glob(os.path.join(data_dir, "images", "*.*")))]
    lbl_files = [f for f in sorted(glob(os.path.join(data_dir, "masks", "*.*")))]
    with open(os.path.join(data_dir, "dataset_split.json")) as fp:
        dataset_split = json.load(fp)

    root_train = out_dir / "train/"
    root_train.mkdir(exist_ok=True, parents=True)
    root_val = out_dir / "validation/"
    root_val.mkdir(exist_ok=True, parents=True)

    for i, (img_file, lbl_file) in enumerate(zip(img_files, lbl_files)):
        img = io.imread(img_file)
        masks = io.imread(lbl_file)

        # reshape to patch
        psize = 256
        img = img[:psize, :psize, :]
        masks = masks[:psize, :psize]

        img = transforms.normalize_img(img, axis=-1).transpose(2, 0, 1)[:3]
        lbls = dynamics.labels_to_flows([masks], device=torch.device("cuda"))[0]
        if Path(img_file).name in dataset_split["training"]:
            tifffile.imwrite(
                root_train / f"FUCCI_data_{i}.tif", data=img, compression="zlib"
            )
            tifffile.imwrite(
                root_train / f"FUCCI_data_{i}_flows.tif", data=lbls, compression="zlib"
            )
            tifffile.imwrite(
                root_train / f"FUCCI_data_{i}_masks.tif",
                data=fastremap.renumber(masks.astype("uint16"))[0],
                compression="zlib",
            )
        elif Path(img_file).name in dataset_split["validation"]:
            tifffile.imwrite(
                root_val / f"FUCCI_data_{i}.tif", data=img, compression="zlib"
            )
            tifffile.imwrite(
                root_val / f"FUCCI_data_{i}_flows.tif", data=lbls, compression="zlib"
            )
            tifffile.imwrite(
                root_val / f"FUCCI_data_{i}_masks.tif",
                data=fastremap.renumber(masks.astype("uint16"))[0],
                compression="zlib",
            )
        else:
            raise RuntimeError(f"Image {Path(img_file).name} not known")


if __name__ == "__main__":
    fucci_data_dir = Path("training_data")
    out_dir = Path("cellpose_sam_dataset")

    print("Converting data")
    convert_fucci_data(fucci_data_dir, out_dir)
