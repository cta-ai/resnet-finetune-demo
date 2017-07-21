#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: helper.py
# Date: 16.06.2017
# Author: Adam Brzeski, Jan Cychnerski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import skimage.transform
from PyQt5 import QtGui


def preprocess(im):

    """
    Preprocesses image array for classifying using ImageNet trained Resnet-152 model
    :param im: RGB, RGBA float-type image or grayscale image
    :return: ready image for passing to a Resnet model
    """

    # Detect invalid images
    if im is None or not hasattr(im, 'shape') or len(im.shape) < 2: return None

    # If grayscale, convert to RGB
    if len(im.shape) == 2:
        im = np.asarray(np.dstack((im, im, im)), dtype=np.uint8)

    # Remove alpha channel, if necessary
    if im.shape[2] == 4:
        im = im[:, :, 0:3]

    if len(im.shape) < 2:
        print("Wrong image shape", im.shape)

    # RGB to BGR
    im = im[:, :, ::-1]

    # Resize and scale values to <0, 255>
    im = skimage.transform.resize(im, (224, 224), mode='constant').astype(np.float32)
    im *= 255

    # Subtract ImageNet mean
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    # Add a dimension
    im = np.expand_dims(im, axis=0)

    return im


def qimage_to_array(image):

    """
    Converts Qt QImage to a numpy image
    :param image: QImage type image
    :return: numpy array image

    """

    if not isinstance(image, QtGui.QImage):
        image = QtGui.QImage(image)

    assert isinstance(image, QtGui.QImage)

    image = image.convertToFormat(QtGui.QImage.Format_ARGB32)

    width = image.width()
    height = image.height()

    ptr = image.constBits()
    ptr.setsize(image.byteCount())

    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr[..., :3].copy()
