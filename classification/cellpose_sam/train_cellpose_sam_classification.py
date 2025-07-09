from cellpose import transforms, dynamics, vit_sam, models, train
from glob import glob
import os
from cellpose import metrics, io
import numpy as np
import json
import torch
import tifffile
import fastremap
from torch import nn
from tqdm import trange, tqdm
from natsort import natsorted
from pathlib import Path


def convert_fucci_phase_data(data_dir, out_dir, n_classes=3):
    # TODO load only from test file ?
    img_files = [f for f in sorted(glob(os.path.join(data_dir, "images", "*.*")))]
    lbl_files = [f for f in sorted(glob(os.path.join(data_dir, "masks", "*.*")))]
    cls_files = [f for f in sorted(glob(os.path.join(data_dir, "classes", "*.*")))]
    with open(os.path.join(data_dir, "dataset_split.json")) as fp:
        dataset_split = json.load(fp)
    
    root_train = out_dir / "train/"
    root_train.mkdir(exist_ok=True, parents=True)
    root_val = out_dir / "validation/"
    root_val.mkdir(exist_ok=True, parents=True)
    
    for i, (img_file, lbl_file, cls_file) in enumerate(zip(img_files, lbl_files, cls_files)):
        img = io.imread(img_file)
        masks = io.imread(lbl_file)

        with open(cls_file) as fp:
            classes = json.load(fp)
        
        # reshape to patch 
        psize = 256
        img = img[:psize, :psize, :]
        masks = masks[:psize, :psize]

        img_shape = img.shape[:2]

	# to store masks
        # one classs added for background
        n_ary_masks = np.zeros((n_classes + 1, img_shape[0], img_shape[1]), "uint16")

        for gt in np.unique(masks):
            if gt == 0:
                continue
            try:
                label_id = classes[gt]
            except KeyError:
                label_id = classes[str(gt)]
            n_ary_masks[label_id][masks == gt] = gt

        img = transforms.normalize_img(img, axis=-1).transpose(2, 0, 1)[:3]
        lbl_all = dynamics.labels_to_flows([masks], device=torch.device("cuda"))[0]
        # img shape changed three lines before
        cp_all = np.zeros(img.shape[1:], "float32")
        print(cp_all.shape)
        for c in range(n_ary_masks.shape[0]):
            print(c)
            cp_all[n_ary_masks[c] > 0] = c + 1

        lbls = np.concatenate((lbl_all[:1], cp_all[np.newaxis,...], lbl_all[1:]), axis=0)
        print(lbls.shape, lbls)
        if Path(img_file).name in dataset_split["training"]:
            tifffile.imwrite(root_train / f"FUCCI_data_{i}.tif", data=img, compression="zlib")
            tifffile.imwrite(root_train / f"FUCCI_data_{i}_flows.tif", data=lbls, compression="zlib")
            tifffile.imwrite(root_train / f"FUCCI_data_{i}_masks.tif", data=fastremap.renumber(masks.astype("uint16"))[0], compression="zlib")
        elif Path(img_file).name in dataset_split["validation"]:
            tifffile.imwrite(root_val / f"FUCCI_data_{i}.tif", data=img, compression="zlib")
            tifffile.imwrite(root_val / f"FUCCI_data_{i}_flows.tif", data=lbls, compression="zlib")
            tifffile.imwrite(root_val / f"FUCCI_data_{i}_masks.tif", data=fastremap.renumber(masks.astype("uint16"))[0], compression="zlib")
        else:
            raise RuntimeError(f"Image {Path(img_file).name} not known")


def initialize_class_net(nclasses=3, device=torch.device("cuda")):
    net = vit_sam.Transformer(rdrop=0.4).to(device)
    # default model
    net.load_model(Path.home() / ".cellpose/models/cpsam", device=device, strict=False)

    # initialize weights for class maps
    ps = 8 # patch size
    nout = 3
    w0 = net.out.weight.data.detach().clone()
    b0 = net.out.bias.data.detach().clone()
    net.out = nn.Conv2d(256, (nout + nclasses + 1) * ps**2, kernel_size=1).to(device)
    # set weights for background map
    i = 0
    net.out.weight.data[i * ps**2 : (i+1) * ps**2] = -0.5*w0[(nout-1) * ps**2 : nout * ps**2]
    net.out.bias.data[i * ps**2 : (i+1) * ps**2] = b0[(nout-1) * ps**2 : nout * ps**2]
    # set weights for maps to 4 nuclei classes
    for i in range(1, nclasses + 1):
        net.out.weight.data[i * ps**2 : (i+1) * ps**2] = 0.5*w0[(nout-1) * ps**2 : nout * ps**2]
        net.out.bias.data[i * ps**2 : (i+1) * ps**2] = b0[(nout-1) * ps**2 : nout * ps**2]
    net.out.weight.data[-(nout * ps**2) : ] = w0
    net.out.bias.data[-(nout * ps**2) : ] = b0
    net.W2 = nn.Parameter(torch.eye((nout + nclasses + 1) * ps**2).reshape((nout + nclasses + 1) * ps**2, nout + nclasses + 1, ps, ps),
                                requires_grad=False)
    net.to(device);
    return net


def train_net(root0, n_classes=3):
    train_files = (root0 / "train").glob("*.tif")
    train_files = natsorted([tf for tf in train_files if "_flows" not in str(tf) and "_masks" not in str(tf)])
    train_files = train_files[::4]
    train_data, test_data = [], []
    print("loading images")
    for i in trange(len(train_files)):
        img = io.imread(train_files[i])
        train_data.append(img)

    print("loading labels")
    train_labels = [io.imread(str(train_files[i])[:-4] + '_flows.tif') for i in trange(len(train_files))]

    pclass = np.zeros((n_classes + 1,))
    pclass_img = np.zeros((len(train_data), n_classes + 1))
    for c in range(n_classes + 1):
        print(c)
        pclass_img[:, c] = np.array([(tl[1] == c).mean() for tl in train_labels])
    pclass = pclass_img.mean(axis=0)
    print("Pclass: ", pclass)

    device = torch.device("cuda")

    net = initialize_class_net(nclasses=n_classes, device=device)

    learning_rate = 5e-5
    weight_decay = 0.1
    batch_size = 8 
    n_epochs = 50
    bsize = 256
    rescale = False
    scale_range = 0.5

    out = train.train_seg(net, train_data=train_data, train_labels=train_labels,
                            learning_rate=learning_rate, weight_decay=weight_decay,
                            batch_size=batch_size, n_epochs=n_epochs,
                            bsize=bsize,
                            nimg_per_epoch=len(train_data),
                            rescale=rescale, scale_range=scale_range,
                            min_train_masks=0,
                            nimg_test_per_epoch=len(test_data),
                            model_name=f"FUCCI_batch_size_{batch_size}",
                            class_weights=1./pclass)


if __name__ == '__main__':
    fucci_data_dir = Path("training_data_tiled_strict_classified_new")
    out_dir = Path("cellpose_sam_dataset")
    
    print("Converting data")
    convert_fucci_phase_data(fucci_data_dir, out_dir)
    print("Training network")
    train_net(out_dir)
