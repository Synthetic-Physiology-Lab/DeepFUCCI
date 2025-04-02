# Data preparation

In the folder `SegmentationProposal`,
we summarise the steps to generate an initial guess
of the segmentation.

We manage the scripts through a `metadata.yml`
file where you need to enter the filename, the channel numbers,
and the expected nuclear diameter (in microns).

## Flat-field correction

If the image has a notable flatfield,
correct it using the `Basic-Flatfield-Correction.py`
script.
Please note that this script produces a BaSiC model
for each channel.
The later scripts will use the models so **make sure that
the models in the folder belong to the data that you process!**

## Using the DAPI-eq approach

In the script `Nuclei-segmentation-max-projection.py`,
a max-projection of the two FUCCI channels is segmented.
The image is denoised by a Gaussian blur and the background is
subtracted using a top-hat filter.
For that, the expected diameter of the nucleus
needs to be added to the metadata YAML file. 

## Using the pre-trained network

Using the script `segment-nuclei.py` prepares three files:
`stardist_labels_{1,2,3}_channel.tif`
They are containing the masks predicted by the three
pre-trained StarDist networks.
**Note: You need to enter the model name and reference
pixel size.**

## Inspect and curate the labels

I opened the labels in Napari by using the `view_in_napari.py`
script. In the example here, I used the StarDist labels.
For other labels, the script has to be adapted.
Then, I duplicated the label layer that seemed
to have the best segmentation and manually curated
this layer.
The manually curated layer is then exported from Napari
to the `dapieq_labels_manual.tif` file. 

The annotation is done on entire videos because the
FUCCI sensor goes dark after division. By switching
between frames, it becomes easier to judge if there is a
nucleus, debris, or an artifact (e.g., from bleedthrough).
However, it makes sense to only annotate single frames
because most of the nuclei look similar.

To facilitate the labeling, there is an option to tile
a single frame in `view_tiled_frame_in_napari.py`.
For example, the tiling will work for 9 tiles like:

1|4|7
2|5|8
3|6|9

Thus, you can open the video and the tiled frame next to each
other and then correct the segmentation.

## Generate training data

Single frames are then exported in training-ready format
through
`extract_frames.py`. 

## Tile data

The data is now still in the original format.
Please refer to the scripts in the `classification`
folder to see an example of a tiling script.
All segmentation masks were also classified prior
to training.
