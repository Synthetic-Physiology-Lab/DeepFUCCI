# Classification


The sensor can distinguish three cell cycle phases:

1. G1 phase (label **1**)
1. G1/S phase (label **2**)
1. S/G2/M phase (label **3**)

Every nuclear label receives a cell cycle label.
These labels are stored in a JSON file.

Details are described in the `DataPreparation` folder.

## Training

See the scripts in the `training_scripts` folder.
They follow mostly the instructions of the StarDist
repository.

## Using the model

Essentially, it boils down to loading the model
and predicting instances

```
model = StarDist2D(None, name="stardist_multiclass", basedir=Path.home() / "models")
label, res = model.predict_instances(img)
```

Examples of the usage are in the validation scripts 
`validation_classifier_*.py`.
The performance of the deep learning
network was compared to the intensity-based classifier
implemented in fucciphase: `check_classification_against_intensity_threshold.py`


Other examples:

* CellMAPtracer: Here, the network is relabeled to match the PIP-FUCCI logic.

## Cellpose-SAM

As an alternative to the StarDist-based network,
we trained an instance of Cellpose-SAM.
Details are provided in the `cellpose_sam` folder.
