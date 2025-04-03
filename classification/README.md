# Classification


The sensor can distinguish three cell cycle phases:

1. G1 phase (label **1**)
1. G1/S phase (label **2**)
1. S/G2/M phase (label **3**)

Every nuclear label receives a cell cycle label.
These labels are stored in a JSON file.


## Data preparation

Prepare the segmentation data as described in the
`segmentation` folder.
You should find a `for_training` folder there.
Make sure that the `fucciphase` package is installed
and run the scripts from the `data_preparation` folder


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


## Training

See the scripts in the `training_scripts` folder.
They follow mostly the instructions of the StarDist
repository.

## Using the model

TODO
