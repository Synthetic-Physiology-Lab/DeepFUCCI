from cellpose import transforms, dynamics, vit_sam, train, models, metrics
from scipy.stats import mode
from glob import glob
import os
from cellpose import io
import numpy as np
import json
import torch
import tifffile
import fastremap
from torch import nn
from tqdm import trange
from natsort import natsorted
from pathlib import Path

logger, log_file = io.logger_setup()


def convert_fucci_phase_data(data_dir, out_dir, n_classes=3):
    img_files = [f for f in sorted(glob(os.path.join(data_dir, "images", "*.*")))]
    lbl_files = [f for f in sorted(glob(os.path.join(data_dir, "masks", "*.*")))]
    cls_files = [f for f in sorted(glob(os.path.join(data_dir, "classes", "*.*")))]
    with open(os.path.join(data_dir, "dataset_split.json")) as fp:
        dataset_split = json.load(fp)

    root_train = out_dir / "train/"
    root_train.mkdir(exist_ok=True, parents=True)
    root_val = out_dir / "validation/"
    root_val.mkdir(exist_ok=True, parents=True)

    for i, (img_file, lbl_file, cls_file) in enumerate(
        zip(img_files, lbl_files, cls_files)
    ):
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
        n_ary_masks = np.zeros((n_classes, img_shape[0], img_shape[1]), "uint16")

        for gt in np.unique(masks):
            if gt == 0:
                continue
            try:
                label_id = classes[gt]
            except KeyError:
                label_id = classes[str(gt)]
            # zero-indexed but classes are one-indexed!
            n_ary_masks[label_id - 1][masks == gt] = gt

        img = transforms.normalize_img(img, axis=-1).transpose(2, 0, 1)[:3]
        lbl_all = dynamics.labels_to_flows([masks], device=torch.device("cuda"))[0]
        # img shape changed three lines before
        cp_all = np.zeros(img.shape[1:], "float32")
        print(cp_all.shape)
        for c in range(n_ary_masks.shape[0]):
            print("C: ", c, " ", np.sum(n_ary_masks[c] > 0))
            cp_all[n_ary_masks[c] > 0] = c + 1

        lbls = np.concatenate(
            (lbl_all[:1], cp_all[np.newaxis, ...], lbl_all[1:]), axis=0
        )
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
            tifffile.imwrite(root_val / f"FUCCI_data_{i}_classes.tif", data=cp_all, compression="zlib")
        else:
            raise RuntimeError(f"Image {Path(img_file).name} not known")


def initialize_class_net(nclasses=3, device=torch.device("cuda")):
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
    # set weights for maps to 4 nuclei classes
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


def train_net(root0, n_classes=3):
    train_files = (root0 / "train").glob("*.tif")
    train_files = natsorted(
        [
            tf
            for tf in train_files
            if "_flows" not in str(tf) and "_masks" not in str(tf)
        ]
    )
    train_files = train_files[::4]
    train_data, test_data = [], []
    print("loading images")
    for i in trange(len(train_files)):
        img = io.imread(train_files[i])
        train_data.append(img)

    print("loading labels")
    train_labels = [
        io.imread(str(train_files[i])[:-4] + "_flows.tif")
        for i in trange(len(train_files))
    ]

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
    n_epochs = 500
    bsize = 256
    rescale = False
    scale_range = 0.5

    _out = train.train_seg(
        net,
        train_data=train_data,
        train_labels=train_labels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        n_epochs=n_epochs,
        bsize=bsize,
        nimg_per_epoch=len(train_data),
        rescale=rescale,
        scale_range=scale_range,
        min_train_masks=0,
        nimg_test_per_epoch=len(test_data),
        model_name=f"FUCCI_batch_size_{batch_size}",
        class_weights=1.0 / pclass,
    )


def test_net(root0, n_classes=3):
    results_dir = Path("results_cellpose_sam")
    results_dir.mkdir(exist_ok=True, parents=True)
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    net = initialize_class_net(nclasses=n_classes, device=torch.device("cuda"))
    net.load_model(
        "models/FUCCI_batch_size_8",
        device=torch.device("cuda"),
        strict=False,
    )
    net.eval()
    model.net = net
    model.net_ortho = None
    img_files = [
        f
        for f in natsorted(root0.glob("*tif"))
        if "_masks" not in str(f)
        and "_flows" not in str(f)
        and "_classes" not in str(f)
    ]
    test_imgs = [io.imread(f) for f in img_files]
    masks_true = [
        io.imread(str(f).replace(".tif", "_masks.tif")).squeeze() for f in img_files
    ]
    classes_true = [
        io.imread(str(f).replace(".tif", "_classes.tif")) for f in img_files
    ]

    masks_pred, flows, styles = model.eval(
        test_imgs,
        diameter=None,
        augment=False,
        bsize=256,
        tile_overlap=0.1,
        batch_size=64,
        flow_threshold=0.4,
        cellprob_threshold=0,
    )

    classes_pred = [s.squeeze().argmax(axis=-1) for s in styles]

    aps_img, errors_img = compute_ap_pq(
        masks_true, classes_true, masks_pred, classes_pred
    )

    print("Errors image: ", np.nanmean(errors_img, axis=0), np.nanmean(errors_img))
    print("APS image: ", np.nanmean(aps_img, axis=0), np.nanmean(aps_img))

    np.save(
        f"{results_dir}/monusac_cellposeSAM.npy",
        {
            "errors": errors_img,
            "aps": aps_img,
            "masks_true": masks_true,
            "masks_pred": masks_pred,
            "classes_true": classes_true,
            "classes": classes_pred,
            "imgs": test_imgs,
            "img_files": img_files,
        },
    )


def compute_ap_pq(masks_true, classes_true, masks_pred, classes_pred, n_classes=3):
    nimg = len(masks_true)
    iou_all, tp_all, fp_all, fn_all = (
        np.zeros((nimg, n_classes), "float32"),
        np.zeros((nimg, n_classes), "int"),
        np.zeros((nimg, n_classes), "int"),
        np.zeros((nimg, n_classes), "int"),
    )
    for i in range(nimg):
        masks_pred0 = masks_pred[i].copy()
        class_true = classes_true[i].copy()
        class0 = classes_pred[i].copy()
        masks_true0 = masks_true[i].copy()
        masks_true0 = fastremap.renumber(masks_true0)[0]

        # remove masks with class 0 (background)
        masks_pred0[class0 == 0] = 0
        masks_pred0 = fastremap.renumber(masks_pred0)[0]

        # class id for each mask is mode of class ids in mask
        cid = np.array(
            [mode(class0[masks_pred0 == j])[0] for j in range(1, masks_pred0.max() + 1)]
        )
        tid = np.array(
            [
                mode(class_true[masks_true0 == j])[0]
                for j in range(1, masks_true0.max() + 1)
            ]
        )

        # match ground truth and predicted masks
        iout, inds = metrics.mask_ious(masks_true0, masks_pred0)
        inds[iout < 0.5] = 0  # keep matches > 0.5 IoU
        # class for matched masks
        cmatch = cid[[ind - 1 for ind in inds if ind != 0]]
        # class for true masks that are matched
        tmatch = tid[inds != 0]
        inds_match = inds[inds != 0]
        for c in range(n_classes):
            # true positive if predicted mask class matches true mask class
            tps = (cmatch == c + 1) * (tmatch == c + 1)
            iou_all[i, c] = (iout[inds != 0] * tps).sum()  # scale by IoU
            tp_all[i, c] = tps.sum()
            # false negative for all missed masks with class == c+1
            fn_all[i, c] = (tid == c + 1).sum() - tps.sum()
            # false positive if predicted mask class == c+1 and does not match true mask class
            not_tp = np.ones(masks_pred0.max(), "bool")
            not_tp[inds_match[tps] - 1] = False
            fp_all[i, c] = (
                cid[not_tp] == c + 1
            ).sum()  # ((cmatch == c+1) * (tmatch != c+1)).sum() +
        assert (fp_all[i].sum() + tp_all[i].sum()) == masks_pred0.max()
        assert (fn_all[i].sum() + tp_all[i].sum()) == masks_true0.max()

    aps_img = tp_all / (tp_all + fp_all + fn_all)
    errors_img = (fp_all + fn_all) / (tp_all + fn_all)

    errors_img[np.isinf(errors_img)] = np.nan
    return aps_img, errors_img

if __name__ == "__main__":
    fucci_data_dir = Path("training_data_tiled_strict_classified_new")
    out_dir = Path("cellpose_sam_dataset")

    print("Converting data")
    convert_fucci_phase_data(fucci_data_dir, out_dir)
    print("Training network")
    train_net(out_dir)
    print("Run validation")
    test_net(out_dir / "validation")
