# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import voc as cfg
from ..box_utils import IoG, decode_new
import sys


class RepulsionLoss(nn.Module):
    def __init__(self, use_gpu=True, sigma=0.):
        super(RepulsionLoss, self).__init__()
        self.use_gpu = use_gpu
        self.variance = cfg['variance']
        self.sigma = sigma

    # TODO: 研究smoothln函数
    def smoothln(self, x, sigma=0.):
        self.sigma = 0
        # 创建sigma张量
        sigma = torch.full(x.shape, sigma)
        if x <= sigma:
            return -torch.log(1 - x + 1e-10)
        else:
            return (x - sigma) / (1 - sigma) - torch.log(1 - sigma)

    def forward(self, loc_data, ground_data, prior_data):
        decoded_boxes = decode_new(loc_data,
                                   Variable(prior_data.data, requires_grad=False),
                                   self.variance)
        # 计算IoG
        iog = IoG(ground_data, decoded_boxes)
        # sigma = 1
        # loss = torch.sum(-torch.log(1-iog+1e-10))
        # sigma = 0
        loss = torch.sum(self.smoothln(iog))
        return loss
