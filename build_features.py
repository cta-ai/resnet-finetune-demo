#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: build_features.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Builds cached Resnet features for train, val and test subsets of Indoor-67 dataset. Cached features are saved to disk
as numpy arrays. Except features, labels infered from paths and paths itself are also saved.

In short, script will create 3 files in each subset directory (train, val, test):
- features-resnet152.npy
- labels-resnet152.npy
- paths-resnet152.json

"""

import glob
import json
import os
import numpy as np
import skimage.io
from keras.models import Model
import helper
from resnet import resnet152


DATA_SUBSETS = [
    os.path.expanduser("~/ml/data/indoor/train"),
    os.path.expanduser("~/ml/data/indoor/val"),
    os.path.expanduser("~/ml/data/indoor/test"),
]
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"
PATHS_FILENAME = "paths-resnet152.json"
WEIGHTS_RESNET = os.path.expanduser("~/ml/models/keras/resnet152/resnet152_weights_tf.h5")
NAMES_TO_IDS = json.load(open("names_to_ids.json"))


# Load Resnet 152 model and construct feature extraction submodel
resnet_model = resnet152.resnet152_model(WEIGHTS_RESNET)
feature_layer = 'avg_pool'
features_model = Model(inputs=resnet_model.input,
                       outputs=resnet_model.get_layer(feature_layer).output)

# For each data subset
for datadir in DATA_SUBSETS:

    features = []
    labels = []
    paths = []
    images_list = glob.glob(datadir + "/*/*.jpg")

    # Process images
    for i, path in enumerate(images_list[:10]):
        try:
            # Load image
            im = skimage.io.imread(path)
            im = helper.preprocess(im)
            if im is None: raise Exception("Could not load image")

            # Run model to get features
            code = features_model.predict(im).flatten()

            # Cache result
            label = NAMES_TO_IDS[os.path.basename(os.path.dirname(path))]
            labels.append(label)
            features.append(code)
            rel_path = os.path.join(*path.split(os.sep)[-2:]).replace("\\", "/")
            paths.append(rel_path)

            # Show progress
            if i % 100 == 0:
                print(i, "/", len(images_list))

        except Exception as e:
            print("Error processing path {}: {}".format(path, e))

    # Save to disk
    np.save(os.path.join(datadir, FEATURES_FILENAME), features)
    np.save(os.path.join(datadir, LABELS_FILENAME), np.uint8(labels))
    with open(os.path.join(datadir, PATHS_FILENAME), mode='w') as f:
        json.dump(paths, f, indent=4)
