# InstanSeg segmentation

To use InstanSeg, please checkout the Github repository:

```
git clone https://github.com/instanseg/instanseg.git
cd instanseg
```

We used InstanSeg at commit cb1bb10e234e7f6e322235786bc1a4a9409ad832
but you should try on the latest commit or release because InstanSeg
is still work in progress.

Copy the FUCCI dataset to `instanseg/Raw_Datasets` and name it `FUCCI_data`.
Copy the `load_custom_dataset.py` to the `instanseg/notebooks` directory
and run it there.

Then execute the following code to train:
```
cd instanseg/scripts
python train.py -data fucci_dataset_1_channels.pth -dim_in 1 -source "[FUCCI_1CH_Dataset]" --experiment_str fucci_1ch
python train.py -data fucci_dataset_2_channels.pth -dim_in 2 -source "[FUCCI_2CH_Dataset]" --experiment_str fucci_2ch
python train.py -data fucci_dataset_3_channels.pth -dim_in 3 -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_3ch
```


Train the channel invariant network:

```
python train.py -ci True -data fucci_dataset_3_channels.pth -source "[FUCCI_3CH_Dataset]" --experiment_str fucci_channel_invariant
```
