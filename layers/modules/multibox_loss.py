# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import voc as cfg
from ..box_utils import match, log_sum_exp
from .repulsion_loss import RepulsionLoss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes)
                loc shape: torch.size(batch_size, num_priors, 4)
                priors shape: torch.size(num_priors, 4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # 之前的网络输出：data
        # 目标框：target
        # 预测框：prediction
        loc_data, conf_data, priors = predictions
        num_batch = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # 对于每个先验框(default boxes)，找到与其相匹配的目标框(ground truth boxes)
        loc_t = torch.Tensor(num_batch, num_priors, 4)
        loc_g1 = torch.Tensor(num_batch, num_priors, 4)
        loc_g2 = torch.Tensor(num_batch, num_priors, 4)
        conf_t = torch.LongTensor(num_batch, num_priors)
        loss_l_repbox = torch.tensor(0.)
        loss_l_repbox_nums = 0
        for idx in range(num_batch):
            predicts = loc_data[idx].data
            # predicts_labels = max_predicts[idx].data
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            tem_a, tem_b = match(self.threshold, predicts, truths, defaults, self.variance, labels, loc_t,
                                 loc_g1, loc_g2, conf_t, idx)
            loss_l_repbox += tem_a
            loss_l_repbox_nums += tem_b
        if self.use_gpu:
            loc_t = loc_t.cuda()
            loc_g1 = loc_g1.cuda()
            loc_g2 = loc_g2.cuda()
            conf_t = conf_t.cuda()
        # 使用Variable进行包装目标框(ground truth boxes)
        loc_t = Variable(loc_t, requires_grad=False)
        loc_g1 = Variable(loc_g1, requires_grad=False)
        loc_g2 = Variable(loc_g2, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        # 位置损失计算，使用Smooth L1，仅针对正样本进行计算
        # 1. 抽取正样本，conf_t > 0 使 conf_t 中每个元素跟0比较得到结果，0为背景类别
        pos = conf_t > 0                                        # shape[b, priors]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # shape[b, num_priors, 4]
        # 2. 抽取正样本进行计算
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loc_g1 = loc_g1[pos_idx].view(-1, 4)
        loc_g2 = loc_g2[pos_idx].view(-1, 4)
        # 原loss函数,
        # Attr: 找正样本(match函数中使用pri和gt做IoU，一个pri对于一个gt），后使用该pri偏移后的predict与对应的gt做loss
        priors = priors.unsqueeze(0).expand_as(loc_data)[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # 新增repulsion loss
        # RepGt: 正样本中，找pri对应的IoU第二大gt，后使用该pri偏移后的predict与该gt计算IoG，在计算loss
        # RepBox: 正样本中计算预测框之间的损失
        repul_loss = RepulsionLoss(sigma=0.)
        loss_l_repul = repul_loss(loc_p, loc_g1, priors)
        loss_l_repul += repul_loss(loc_p, loc_g2, priors)

        # 置信度损失计算
        # 1. Hard Negative Mining, 需要将所有batch的图片一起找到难负例
        batch_conf = conf_data.view(-1, self.num_classes)                            # shape[b*priors, classes]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # shape[b*priors, classes]
        loss_c = loss_c.view(num_batch, -1)                                          # shape[b, priors]
        # 2. 把正样本排除，剩下的就全是负样本，可以进行抽样，这里使用pos即可，不用pos_idx
        loss_c[pos] = 0
        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 抽取负样本
        num_pos = pos.long().sum(1, keepdim=True)  # shape[batch, 1]
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # 3. Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # shape[b, priors, ]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # for conf_data
        # 3. 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算conf交叉熵: -Sum( (P_i * log(Q_i)) )
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # α默认设置为1
        N = num_pos.data.sum()
        loss_l /= N
        loss_l_repul = loss_l_repul / N + loss_l_repbox / (loss_l_repbox_nums + 1e-10)
        loss_c /= N
        return loss_l, loss_l_repul, loss_c
