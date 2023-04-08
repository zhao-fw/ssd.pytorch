from .config import *
from .voc0712 import VOCDetection, VOCAnnotationTransform
import torch
import cv2
import numpy as np


def detection_collate(batch):
    """手动将抽取出的样本堆叠起来的函数
    当一次取出batch个图片时，将所有图片数据组合到一个变量中（本来是list）
    只有在train时，需要batch，需要使用该函数

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    """处理图片：resize到(size, size), 减去平均值
    在eval、test中使用
    """
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
