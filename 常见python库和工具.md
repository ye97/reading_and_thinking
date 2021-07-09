---
title: 常见python库和工具
categories:
  - 开发
  - python
  - 工具

description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-07-09 10:29:49
urlname:
tags:
---





# cuda

CUDA(ComputeUnified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。 CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。



# Cudann

 cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。

# 安装cudann

 从官方安装指南可以看出，只要把cuDNN文件复制到CUDA的对应文件夹里就可以，即是所谓插入式设计，把cuDNN数据库添加CUDA里，cuDNN是CUDA的扩展计算库，不会对CUDA造成其他影响。

卸载删除相应的文件即可。

# cuda和cudann关系

CUDA看作是一个工作台，上面配有很多工具，如锤子、螺丝刀等。cuDNN是基于CUDA的深度学习GPU加速库，有了它才能在GPU上完成深度学习的计算。它就相当于工作的工具，比如它就是个扳手。但是CUDA这个工作台买来的时候，并没有送扳手。想要在CUDA上运行深度神经网络，就要安装cuDNN，就像你想要拧个螺帽就要把扳手买回来。这样才能使GPU进行深度神经网络的工作，工作速度相较CPU快很多。

# torchvision

安装pytorch时，torchvision独立于torch。torchvision包由流行的数据集（torchvision.datasets）、模型架构(torchvision.models)和用于计算机视觉的常见图像转换组成t(torchvision.transforms)。

1）数据库

*   [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)
*   [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
*   [KMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist)
*   [EMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)
*   [COCO](https://pytorch.org/docs/stable/torchvision/datasets.html#coco)
    *   [Captions](https://pytorch.org/docs/stable/torchvision/datasets.html#captions)
    *   [Detection](https://pytorch.org/docs/stable/torchvision/datasets.html#detection)
*   [LSUN](https://pytorch.org/docs/stable/torchvision/datasets.html#lsun)
*   [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
*   [DatasetFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)
*   [Imagenet-12](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet-12)
*   [CIFAR](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
*   [STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#stl10)
*   [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn)
*   [PhotoTour](https://pytorch.org/docs/stable/torchvision/datasets.html#phototour)
*   [SBU](https://pytorch.org/docs/stable/torchvision/datasets.html#sbu)
*   [Flickr](https://pytorch.org/docs/stable/torchvision/datasets.html#flickr)
*   [VOC](https://pytorch.org/docs/stable/torchvision/datasets.html#voc)
*   [Cityscapes](https://pytorch.org/docs/stable/torchvision/datasets.html#cityscapes)

使用torchvision.datasets中的数据集

```plain
import torchvision
mnist = torchvision.datasets.MNIST("path/to/mnist/", train=True, transform=transforms, target_transform=None, download=False)
```

2）模型框架

*   [Alexnet](https://pytorch.org/docs/stable/torchvision/models.html#id1)
*   [VGG](https://pytorch.org/docs/stable/torchvision/models.html#id2)
*   [ResNet](https://pytorch.org/docs/stable/torchvision/models.html#id3)
*   [SqueezeNet](https://pytorch.org/docs/stable/torchvision/models.html#id4)
*   [DenseNet](https://pytorch.org/docs/stable/torchvision/models.html#id5)
*   [Inception v3](https://pytorch.org/docs/stable/torchvision/models.html#inception-v3)

使用torchvision.models中的模型

```plain
import torchvision
vgg16 = torchvision.models.vgg16(pretrained=True)
```

3)变换操作

*   [Transforms on PIL Image](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image)
*   [Transforms on torch.\*Tensor](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)
*   [Conversion Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#conversion-transforms)
*   [Generic Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#generic-transforms)
*   [Functional Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms)

Transforms on PIL Image中常用的有操作：

```plain
torchvision.transforms.CenterCrop(size)
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
torchvision.transforms.RandomHorizontalFlip(p=0.5)
torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
torchvision.transforms.Resize(size, interpolation=2)
torchvision.transforms.Scale(*args, **kwargs)
#还有更多再次不列出，
```

[更多详见](https://pytorch.org/vision/stable/transforms.html)

使用torchvision.transforms中的变换

```plain
transfrom = torchvision.transforms.CenterCrop(224)
```

