from stardist.models import StarDist2D
import csbdeep
import yaml
from aicsimageio import AICSImage
import numpy as np
from skimage.io import imsave
import json


def class_from_res(res):
    # cast to int (for json)
    cls_dict = dict((int(i+1),int(c)) for i,c in enumerate(res['class_id']))
    return cls_dict

scale = 0.65 / 0.3
print("Scaling with factor: ", scale)

model = StarDist2D(None, name='stardist_multiclass_2_channel_new_data', basedir='/home/julius/models')

with open("metadata.yml", "r") as metadatafile:
    metadata = yaml.load(metadatafile, yaml.BaseLoader)
channels = metadata["channels"]
filename = metadata["filename"]
img_stream = AICSImage(filename)

T = -1
img_tubulin = img_stream.get_image_data("YX", S=int(channels["tubulin"]), T=T)
img_cyan = img_stream.get_image_data("YX", S=int(channels["cyan"]), T=T)
img_magenta = img_stream.get_image_data("YX", S=int(channels["magenta"]), T=T)

# normalize image
img_tubulin = csbdeep.utils.normalize(img_tubulin, pmin=1, clip=True)
img_cyan = csbdeep.utils.normalize(img_cyan, pmin=1, clip=True)
img_magenta = csbdeep.utils.normalize(img_magenta, pmin=1, clip=True)

# 2 channel model
labels, res = model.predict_instances(np.moveaxis(np.stack([img_cyan, img_magenta]), 0, -1), scale=scale)

# the class object ids are stored in the 'results' dict and correspond to the label ids in increasing order 

print(class_from_res(res))
imsave("labels_predicted_channel.tif", labels)
cls_dict = class_from_res(res)
print(cls_dict)
with open("labels_predicted_channel.json", "w") as fp:
    json.dump(cls_dict, fp)
