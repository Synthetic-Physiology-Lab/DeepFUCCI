from aicsimageio import AICSImage
import numpy as np
from skimage.io import imsave
from stardist.models import StarDist2D
from tqdm import tqdm
import csbdeep

filename = "MJC004_Pos1_Control_4C_FUCCI.tif"
img_stream = AICSImage(filename)
labels_2d = np.zeros(shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16)

model_2d = StarDist2D(None, name='stardist_2_channel_grid', basedir='models')
for T in tqdm(range(img_stream.dims.T)):
    img_cyan = img_stream.get_image_data("YX", C=0, T=T)
    img_magenta = img_stream.get_image_data("YX", C=1, T=T)

    img_cyan = csbdeep.utils.normalize(img_cyan, pmin=1, clip=True)
    img_magenta = csbdeep.utils.normalize(img_magenta, pmin=1, clip=True)

    # 2 channel model
    labels, details = model_2d.predict_instances(np.moveaxis(np.stack([img_cyan, img_magenta]), 0, -1), scale=2.0)
    labels_2d[T] = labels[:]

imsave("stardist_labels_2_channel.tif", labels_2d)
