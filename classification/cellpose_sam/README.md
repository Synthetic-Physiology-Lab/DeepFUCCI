# Cellpose-SAM for classification

Following the recent Cellpose-SAM preprint, I adapted
the script provided on
[Github](https://github.com/MouseLand/cellpose/blob/main/paper/cpsam/semantic.py)
so that it worked with our DeepFUCCI data.

## Training

You can run `train_cellpose_sam_classification.py`.
Make sure that the path to the training data directory
is correctly specified at the end of the script.

To visually validate the trained model, have a look at
`plot_cellpose_sam_with_napari.py`.


## Using it

Run the model as in the script `predict_cellpose.py`.
