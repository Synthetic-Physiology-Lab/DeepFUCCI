from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from skimage.io import imsave

# the movie was dark after this slide
cutoff_slide = 167
image = AICSImage("MJC004 Pos1 Control 4C.tif")
image_data = image.get_image_data("TCYX", T=range(cutoff_slide), C=[0, 1])
masks = image.get_image_data("TYX", T=range(cutoff_slide), C=3)

imsave("cellpose_masks.tif", masks)

OmeTiffWriter.save(
    image_data,
    "MJC004_Pos1_Control_4C_FUCCI.tif",
    dim_order="TCYX",
    physical_pixel_sizes=image.physical_pixel_sizes,
)

