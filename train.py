#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: train.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Performs training of a single fully-connected classifier layer on a cached set of feature vectors prepared with
build_features.py. Trained model is saved to classifier_weights.h5.
"""

import os
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


TRAIN_DIR = os.path.expanduser("~/ml/data/indoor/train")
VAL_DIR = os.path.expanduser("~/ml/data/indoor/val")
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"
WEIGHTS_CLASSIFIER = "classifier_weights.h5"


# Load train data
train_features = np.load(os.path.join(TRAIN_DIR, FEATURES_FILENAME))
train_labels = np.load(os.path.join(TRAIN_DIR, LABELS_FILENAME))
train_labels = keras.utils.np_utils.to_categorical(train_labels)

# Load val data
val_features = np.load(os.path.join(VAL_DIR, FEATURES_FILENAME))
val_labels = np.load(os.path.join(VAL_DIR, LABELS_FILENAME))
val_labels = keras.utils.np_utils.to_categorical(val_labels)

# Build softmax model
classifier_model = Sequential()
classifier_model.add(Dense(67, activation='softmax',
                           kernel_initializer='TruncatedNormal',
                           bias_initializer='zeros',
                           input_shape=train_features.shape[1:]))

# Define optimizer and compile
opt = SGD(lr=0.1)
classifier_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare callbacks
lr_decay = ReduceLROnPlateau(factor=0.9, patience=1, verbose=1)
checkpointer = ModelCheckpoint(filepath=WEIGHTS_CLASSIFIER, save_best_only=True, verbose=1)

# Train
classifier_model.fit(train_features, train_labels,
                     epochs=50,
                     batch_size=256,
                     validation_data=(val_features, val_labels),
                     callbacks=[lr_decay, checkpointer])
