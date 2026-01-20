import os
from instanseg import InstanSeg
from aicsimageio import AICSImage
import numpy as np
from skimage.io import imsave
from tqdm import tqdm


os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.expanduser("~/Documents/github/instanseg/instanseg/bioimageio_models")
instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1)


# Channels
channels = {"cyan": 0, "magenta": 1}

filename = "MJC004_Pos1_Control_4C_FUCCI.tif"

img_stream = AICSImage(filename)
labels_2d = np.zeros(shape=(img_stream.dims.T, img_stream.dims.Y, img_stream.dims.X), dtype=np.uint16)

pixel_size = img_stream.physical_pixel_sizes.X
print(pixel_size)

for T in tqdm(range(img_stream.dims.T)):
    img_cyan = img_stream.get_image_data("YX", C=int(channels["cyan"]), T=T)
    img_magenta = img_stream.get_image_data("YX", C=int(channels["magenta"]), T=T)

    # 2 channel model
    labels = instanseg_fluorescence.eval_small_image(image=np.stack([img_cyan, img_magenta]), pixel_size=pixel_size, return_image_tensor=False, target="nuclei")
    labels_2d[T] = labels[:]

imsave("instanseg_pretrained_labels_2_channel.tif", labels_2d)
