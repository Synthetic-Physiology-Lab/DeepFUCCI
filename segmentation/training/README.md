## Folder structure

Organise the folders as:

```
training_data
training_1_channel_stardist
training_2_channels_stardist
training_3_channels_stardist
```

Copy the training scripts into the respective folder,
i.e. `training_1_channel.py` into `training_1_channel_stardist`

## Training

Run the Python scripts in the `training_*` folders.
A successful training will produce a `models`
folder containing a folder called `stardist`.
The `stardist` folder can be copied elsewhere.
To use the models on your computer with the
scripts provided in the other directories,
copy the `stardist` folder (without the logs to
save disk space) to the `models` folder
in your home directory.
Name it, for example, `stardist_1_channel_latest`.
The logs in the `stardist` folder can be viewed
in `TensorBoard`.

After a successful training, you should have the
following folder structure:

```
training_1_channel_stardist/models/stardist
training_2_channels_stardist/models/stardist
training_3_channels_stardist/models/stardist
```

and you should have prepared the following folder
structure in your home directory:

```
$HOME/models/stardist_1_channel_latest
$HOME/models/stardist_2_channel_latest
$HOME/models/stardist_3_channel_latest
```


## Adjusting the scripts for your own dataset

The scripts mostly differ in the way the channels are read in.
If your dataset has a different number of channels,
change the `n_channel` variable.
If the order of your dataset is different, adjust the mapping
in the `normalize` function call.
