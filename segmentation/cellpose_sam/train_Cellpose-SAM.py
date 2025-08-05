from cellpose import models, core, io
from pathlib import Path
from cellpose import train

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu() is False:
  raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)


# Input directory with your images (if you have them, otherwise use sample images):

train_dir = "../training_data_for_cellpose"
if not Path(train_dir).exists():
  raise FileNotFoundError("directory does not exist")

test_dir = None # optionally you can specify a directory with test files

# *** change to your mask extension ***
masks_ext = "_masks"

# list all files
files = [f for f in Path(train_dir).glob("*") if "_masks" not in f.name and "_flows" not in f.name and "_seg" not in f.name]

if(len(files)==0):
  raise FileNotFoundError("no files found, did you specify the correct folder and extension?")
else:
  print(f"{len(files)} files in folder:")

for f in files:
  print(f.name)


# ## Train new model

model_name = "FUCCI_cpsam"

# default training params
n_epochs = 200
learning_rate = 1e-5
weight_decay = 0.1
batch_size = 4

# get files
output = io.load_train_test_data(train_dir, test_dir, mask_filter=masks_ext)
train_data, train_labels, _, test_data, test_labels, _ = output
# (not passing test data into function to speed up training)

new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=train_data,
                                                            train_labels=train_labels,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=max(2, len(train_data)), # can change this
                                                            model_name=model_name)


print(new_model_path)
