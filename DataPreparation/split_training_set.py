import numpy as np
from glob import glob
import os
import json

training_data_dir = "training_data"

X = sorted(glob(f"{training_data_dir}/images/*.tif"))

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val = [os.path.basename(X[i]) for i in ind_val]
X_trn = [os.path.basename(X[i]) for i in ind_train]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

with open(f"{training_data_dir}/dataset_split.json", "w") as fp:
    json.dump({"training": X_trn, "validation": X_val}, fp)
