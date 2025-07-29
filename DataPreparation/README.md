# Data preparation

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

```
1 | 4 | 7
2 | 5 | 8
3 | 6 | 9
```

Thus, you can open the video and the tiled frame next to each
other and then correct the segmentation.

## Generate training data

Single frames are then exported in training-ready format
through
`extract_frames.py`. 
Now, there should be a folder `for_training`.

## Classify data

Make sure that the `fucciphase` package is installed
and run

```
python relabel_and_classify_test_data.py
```

This creates a folder `for_training_relabeled_classified`.
Copy the `check_classifications.py` script to this folder
and run it.
It opens Napari and shows the image, segmentation masks, and proposal
for the classification.
Check if all labels correct. If not, open the json-file
that is shown in the current view (it can be found in
`for_training_relabeled_classified/classes`) and
correct the entry for the segmentation masks (for that,
you can select the label in the `label` layer of Napari,
which will give you the label ID).
The name of the file that is currently opened is printed
in the command line.
When you are done with the file, hit enter in the command line
 and the next file will open.
**Note: A new points layer is opened, delete the old one!**
After the last file, nothing will happen and can you close napari.

Copy the files into the general training data folder.
Here, it is called `training_data_relabeled_classified`.
It holds all annotated data that is not yet cropped or tiled.
In a next step, the data is tiled (here to 256x256 pixels).
The number of nuclei per crop are counted. If there are less
than 4 nuclei (nuclei touching the border are not counted),
the crop is discarded.
The script for tiling and filtering is called:

```
python tile_training_data.py
```
