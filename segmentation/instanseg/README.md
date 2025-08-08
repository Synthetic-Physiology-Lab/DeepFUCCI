# InstanSeg segmentation

To use InstanSeg, please install it as indicated in the requirements
file specified in the repository.

Check out the folder structure of the InstanSeg repository: https://github.com/instanseg/instanseg/tree/v0.1.0
The following description follows this.
Copy the FUCCI dataset to `instanseg/Raw_Datasets` and name it `FUCCI_data`.
Copy the `load_custom_dataset.py` to the `instanseg/notebooks` directory
and run it there.
Check also the InstanSeg github repository for more details.


Then execute the following code to train:
```
cd instanseg/scripts
python train.py -data fucci_1_channels_dataset.pth -dim_in 1 -source "[FUCCI_1CH_Dataset]" --experiment_str fucci_1ch -pixel_size 0.335
python train.py -data fucci_2_channels_dataset.pth -dim_in 2 -source "[FUCCI_2CH_Dataset]" --experiment_str fucci_2ch -pixel_size 0.335
python train.py -data fucci_3_channels_dataset.pth -dim_in 3 -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_3ch -pixel_size 0.335
```


Train the channel invariant network:

```
python train.py -ci True -data fucci_3_channels_dataset.pth -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_channel_invariant
```

## Test the models

Set proper environment variables:

```
export INSTANSEG_MODEL_PATH=~/Documents/github/instanseg/instanseg/models
export INSTANSEG_BIOIMAGEIO_PATH=~/Documents/github/instanseg/instanseg/bioimageio_models
export INSTANSEG_TORCHSCRIPT_PATH=~/Documents/github/instanseg/instanseg/torchscripts
export EXAMPLE_IMAGE_PATH=~/Documents/github/instanseg/instanseg/examples
```

Run the two scripts to test the models and obtain metrics like the F1 score and export the model:

```

python test.py --model_folder fucci_3ch -set Validation --optimize_hyperparameters True --dataset fucci_3_channels
python test.py --model_folder fucci_3ch -set Test --params best_params --dataset fucci_3_channels -export_to_bioimageio True -export_to_torchscript True
```

## Validate the models on the DeepFUCCI dataset

Check the scripts
`validate_all_networks.py`
`validate_channel_invariant_network.py`
