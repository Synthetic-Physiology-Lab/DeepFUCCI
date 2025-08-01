# InstanSeg segmentation

To use InstanSeg, please install it as indicated in the requirements
file specified in the repository.

Copy the FUCCI dataset to `instanseg/Raw_Datasets` and name it `FUCCI_data`.
Copy the `load_custom_dataset.py` to the `instanseg/notebooks` directory
and run it there.
Check also the InstanSeg github repository for more details.


Then execute the following code to train:
```
cd instanseg/scripts
python train.py -data fucci_1_channels_dataset.pth -dim_in 1 -source "[FUCCI_1CH_Dataset]" --experiment_str fucci_1ch
python train.py -data fucci_2_channels_dataset.pth -dim_in 2 -source "[FUCCI_2CH_Dataset]" --experiment_str fucci_2ch
python train.py -data fucci_3_channels_dataset.pth -dim_in 3 -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_3ch
```


Train the channel invariant network:

```
python train.py -ci True -data fucci_3_channels_dataset.pth -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_channel_invariant
```

## Test the models

Run the two scripts to test the models and obtain metrics like the F1 score:

```
python test.py --model_folder fucci_3ch -set Validation --optimize_hyperparameters True --dataset fucci_3_channels
python test.py --model_folder fucci_3ch -set Test --params best_params --dataset fucci_3_channels
```
