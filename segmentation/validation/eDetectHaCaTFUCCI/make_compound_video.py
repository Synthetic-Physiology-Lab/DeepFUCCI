from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import glob
import numpy as np

# get all green files
all_green_files = glob.glob("G/*.tif")
all_green_files = sorted(all_green_files)
for idx, file in enumerate(all_green_files):
    filename = file.replace("G/", "").replace("c1.tif", "")
    green_channel = f"G/{filename}c1.tif"
    if green_channel != file:
        raise ValueError(f"Distilled wrong file name {green_channel} from {file}")
    red_channel = f"R/{filename}c2.tif"
    yellow_channel = f"Y/{filename}c3.tif"

    green_image = AICSImage(green_channel)
    physical_pixel_sizes = green_image.physical_pixel_sizes
    green_image = green_image.get_image_data("YX")
    red_image = AICSImage(red_channel)
    red_image = red_image.get_image_data("YX")
    yellow_image = AICSImage(yellow_channel)
    yellow_image = yellow_image.get_image_data("YX")

    if idx == 0:
        all_frames = np.zeros(shape=(3, green_image.shape[0], green_image.shape[1]))
        all_frames[0, :] = red_image[:]
        all_frames[1, :] = green_image[:]
        all_frames[2, :] = yellow_image[:]
        all_frames = np.expand_dims(all_frames, axis=0)
    else:
        stack = np.stack((red_image, green_image, yellow_image))
        stack = np.expand_dims(stack, axis=0)
        print(stack.shape)
        all_frames = np.append(all_frames, stack, axis=0)
    print(all_frames.shape)

OmeTiffWriter.save(
    all_frames,
    "merged.ome.tif",
    dim_order="TCYX",
    channel_names=[["red", "green", "yellow"]],
    physical_pixel_sizes=physical_pixel_sizes,
)
