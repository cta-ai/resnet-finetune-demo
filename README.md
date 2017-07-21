# resnet-finetune-demo

<a href="url">
<img align="right" height="250" src="https://lh4.googleusercontent.com/j8uIyHQKU8ktm8REITH3B0c8eLvQFps9_zv8S3SAvDhz1wftiQANufKx6Et0gekqUUyiQkbX=w3076-h1670">
</a>

A simple experiment with finetuning Resnet-152 in Keras for classifying indoor places images from [MIT Indoor-67](http://web.mit.edu/torralba/www/indoor.html) dataset. This method achieves 73% accuracy on the test set.

The code is built on Keras 2.0 (ver. 2.0.4) on TensorFlow backend (ver. 1.2.0-rc2) using Python 3.6.

To run this experiment you should download the data package which is available [here](https://drive.google.com/uc?export=download&id=0B7mi_caywPhZalQ1QmZ2TU9fbjg). Unpack it in your home directory for compliance with the code. The data package consists of several elements:
- [Indoor-67](http://web.mit.edu/torralba/www/indoor.html) images, split into train, val and test subsets
- ImageNet-trained Resnet-152 weights, acquired from [here](https://github.com/flyyufelix/cnn_finetune)
- Text file mapping ImageNet ids to class names, acquired from [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
- Cached features from Resnet-152 model and labels for images in train, val and test subsets


#### Method summary

This method simply freezes the entire Resnet-152 model and retrains the last fully-connected classification layer from scratch. It achieves 73% accuracy, which is only 6% less than state-of-the-art as for 2016 (see the chart below).

<p align="left"><a href="url">
<img height="250" src="https://drive.google.com/uc?export=download&id=0B7mi_caywPhZTUt5M1JCTzFKUW8">
</a></p>


##### References

[1] Quattoni, Ariadna, and Antonio Torralba. "Recognizing indoor scenes." Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.

[2] Zhou, Bolei, et al. "Learning deep features for scene recognition using places database." Advances in neural information processing systems. 2014.

[3] Khan, Salman H., et al. "A discriminative representation of convolutional features for indoor scene recognition." IEEE Transactions on Image Processing 25.7 (2016): 3372-3383.

[4] Herranz, Luis, Shuqiang Jiang, and Xiangyang Li. "Scene recognition with CNNs: objects, scales and dataset bias." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.


#### Modules outline:
- resnet-demo.py - demo app testing ImageNet trained Resnet-152 on images from clipboard
- build_features.py - creates a cache of Resnet features over the dataset
- train.py - builds fully-connected classification layer and trains it over the cached features
- test.py - test the trained classifier
- test-demo.py - demo app classifying test images one by one and showing outputs
- helper.py - some helping functions
- resnet/resnet152.py - Resnet-152 model acquired from [here](https://github.com/flyyufelix/cnn_finetune)
