#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: test.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Tests the trained classifier on the cached features and labels of the test set.
"""

import os
from collections import defaultdict
import numpy as np
from keras.layers import Dense
from keras.models import Sequential


WEIGHTS_CLASSIFIER = "classifier_weights.h5"
TEST_DIR = os.path.expanduser("~/ml/data/indoor/test")
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"


# Load test data
test_features = np.load(os.path.join(TEST_DIR, FEATURES_FILENAME))
test_labels = np.load(os.path.join(TEST_DIR, LABELS_FILENAME))

# Load top layer classifier model
classifier_model = Sequential()
classifier_model.add(Dense(67, activation='softmax', input_shape=test_features.shape[1:]))
classifier_model.load_weights(WEIGHTS_CLASSIFIER)

# Classify the test set, count correct answers
all_count = defaultdict(int)
correct_count = defaultdict(int)
for code, gt in zip(test_features, test_labels):

    code = np.expand_dims(code, axis=0)
    prediction = classifier_model.predict(code)
    result = np.argmax(prediction)

    all_count[gt] += 1
    if gt == result:
        correct_count[gt] += 1

# Calculate accuracies
print("Average per class acc:",
      np.mean([correct_count[classid] / all_count[classid] for classid in all_count.keys()]))
print("Overall acc:",
      sum(correct_count.values()) / sum(all_count.values()))
