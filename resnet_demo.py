#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: resnet_demo.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Simple Resnet-152 demo script, allowing you test the model on images by simply copying them into clipboard.
"""

import os
import sys
import numpy as np
import skimage.io
import skimage.transform
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from matplotlib import pyplot as plt
import helper
from resnet import resnet152


WEIGHTS = os.path.expanduser("~/ml/models/keras/resnet152/resnet152_weights_tf.h5")
CLSID_TO_HUMAN = os.path.expanduser('~/ml/models/keras/resnet152/imagenet1000_clsid_to_human.txt')


model = resnet152.resnet152_model(WEIGHTS)
with open(CLSID_TO_HUMAN, 'r') as f:
    id2label = eval(f.read())


@pyqtSlot()
def clipboard_changed():
    clipboard = QApplication.clipboard()
    try:
        if clipboard.mimeData().hasImage():
            image = clipboard.pixmap()
            image = helper.qimage_to_array(image)
            clipboard.clear()
            if image.shape:
                print("----------------------------------------------------------------------------------------")
                print("Processing: image from clipboard")
                process(image)
                skimage.io.imshow(image[:,:,::-1])
                plt.show()
                print("\nWaiting for an image...")

    except Exception as e:
        print("ERROR:", e)


def process(im):
    # Convert to RGB (to comply with helper.preprocess())
    im = im[:, :, ::-1]

    # Preprocess
    im = helper.preprocess(im)

    # Predict
    prediction = model.predict(im)

    # Print results
    prediction = prediction.flatten()
    top_idx = np.argsort(prediction)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print("{}. {:.2f} {}".format(i+1, prediction[idx], id2label[idx]))


app = QApplication(sys.argv)
clipboard = app.clipboard()
clipboard.dataChanged.connect(clipboard_changed)
Form = QWidget()
Form.show()
print("Model ready. You can now copy your test image to clipboard.")
print("Waiting for an image...")
sys.exit(app.exec_())
