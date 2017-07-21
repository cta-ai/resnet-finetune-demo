#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: test_demo.py
# Date: 16.06.2017
# Author: Adam Brzeski, CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Loads test images and classifies them one by one with showing the results and the image itself.
"""

import glob
import json
import os
import random
import numpy as np
import skimage.io
from keras.layers import Dense
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
import helper
from resnet import resnet152


WEIGHTS_RESNET = os.path.expanduser("~/ml/models/keras/resnet152/resnet152_weights_tf.h5")
WEIGHTS_CLASSIFIER = "classifier_weights.h5"
IDS_TO_NAMES = json.load(open("ids_to_names.json"))


# Load Resnet 152 model and construct feature extraction submodel
resnet_model = resnet152.resnet152_model(WEIGHTS_RESNET)
feature_layer = 'avg_pool'
feature_vector_size = int(resnet_model.get_layer(feature_layer).output.shape[-1])
features_model = Model(inputs=resnet_model.input,
                       outputs=resnet_model.get_layer(feature_layer).output)

# Load classifier model
classifier_model = Sequential()
classifier_model.add(Dense(67, activation='softmax', input_shape=[feature_vector_size]))
classifier_model.load_weights(WEIGHTS_CLASSIFIER)

# Load test images
paths = glob.glob(os.path.expanduser("~/ml/data/indoor/test/*/*.jpg"))
random.shuffle(paths)

# Classify images
for path in paths:
    print("----------------------------------------------------------------------------------------")
    print("Classifying image: ", path)

    # Load and preprocess image
    im = skimage.io.imread(path)
    transformed = helper.preprocess(im)
    if transformed is None: continue

    # Classify
    code = features_model.predict(transformed).reshape(1, -1)
    prediction = classifier_model.predict(code)

    # Print result
    prediction = prediction.flatten()
    top_idx = np.argsort(prediction)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print("{}. {:.2f} {}".format(i + 1, prediction[idx], IDS_TO_NAMES[str(idx)]))

    # Show image
    skimage.io.imshow(im)
    plt.show()
