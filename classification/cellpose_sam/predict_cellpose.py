from cellpose import models, vit_sam, io
import torch
from torch import nn
from pathlib import Path
from glob import glob
from skimage.io import imsave


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


n_classes = 3
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
net = initialize_class_net(nclasses=n_classes, device=torch.device("cuda"))
net.load_model(
    "../models/FUCCI_batch_size_8",
    device=torch.device("cuda"),
    strict=False,
)
net.eval()
model.net = net
model.net_ortho = None

X = sorted(glob("images/*.tif"))
img_files = [X[0]]
test_imgs = [io.imread(f) for f in img_files]
masks_true = [io.imread(str(f).replace("images", "masks")).squeeze() for f in img_files]

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

imsave("masks_cp_sam.tif", masks_pred[0], compression="zlib")
imsave("classes_cp_sam.tif", classes_pred[0], compression="zlib")
