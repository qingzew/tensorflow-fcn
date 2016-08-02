# tensorflow-fcn
This is a Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The network can be applied directly or finetuned using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Caffe-Tensorflow is required to run this. <b>The .npy file for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> however need to be downloaded before using this needwork.</b>

For Pascal VOC, something has been added:
1. a loss function has been defined in `fnc8_vgg.py`
2. a reader for VOC has been implemented in 'reader.py'
3. a trainer has been implemented in 'train.py'

## Usage

for training:
`python train.py`

Be aware, that `num_classes` influences the way `score_fr` (the original `fc8` layer) is initialized. For finetuning I recommend using the option `random_init_fc8=True`. 

## Content

Currently the following Models are provided:

- FCN32
- FCN16
- FCN8

## Remark

The deconv layer of tensorflow allows to provide a shape. The crop layer of the original implementation is therefore not needed.

I have slightly altered the naming of the upscore layer.

#### Field of View

The receptive field (also known as or `field of view`) of the provided model is: 

`( ( ( ( ( 7 ) * 2 + 6 ) * 2 + 6 ) * 2 + 6 ) * 2 + 4 ) * 2 + 4 = 404`

## Predecessors

Weights were generated using [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). The VGG implementation is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and numpy loading is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). You do not need any of the above cited code to run the model, not do you need caffe.

## Install

Installing matplotlib from pip requires the following packages to be installed `libpng-dev`, `libjpeg8-dev`, `libfreetype6-dev` and `pkg-config`. On Debian, Linux Mint and Ubuntu Systems type:

`sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config` <br>
`pip install -r requirements.txt`
