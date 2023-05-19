# -*- coding: utf-8 -*-
from __future__ import division
import torch
import time
import random


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # 计算每个目标框a与每个先验框b之间的交集的面积大小
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 同上，计算并集
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def IoG(box_a, box_b):
    """Compute the IoG of two sets of boxes.
    E.g.:
        A ∩ B / A = A ∩ B / area(A)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_objects,4]
    Return:
        IoG: (tensor) Shape: [num_objects]
    """
    inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
    inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
    inter_xmax = torch.min(box_a[:, 2], box_b[:, 2])
    inter_ymax = torch.min(box_a[:, 3], box_b[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    return I / G

def match(threshold, predicts, truths, priors, variances, labels, loc_t, loc_g1, loc_g2, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    new update: Match each predict box with the second largest target
    Args:
        阈值 threshold: (float) The overlap threshold used when mathing boxes.
        所有预测框 predicts: (tensor) encoded predict boxes, Shape: [num_obj, num_priors].
        所有真实框 truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        所有先验框 priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        方差 variances: (tensor) Variances corresponding to each prior coord, Shape: [num_priors, 4].
        所属的类别 labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled with encoded location targets.
        loc_g: (tensor) Tensor to be filled with decoded second largest location targets.
        conf_t: (tensor) Tensor to be filled with matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # 计算gt set与pri set的iou, Shape: [gt_nums, pri_nums]
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # 为每个 gt框 找到最合适的 pri框, Shape: [gt_nums, 1]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # 为每个 pri框 找到最合适的 gt框, Shape: [1, pri_nums]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # 降维
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # 保证每个 gt框 与 一个pri框 匹配，设iou为固定值
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 设匹配的框id（由于一个pri预测一个gt，防止出现pri与两个gt的iou都很大导致漏匹配）
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 每个 pri框 对应的 gt框 的位置信息, Shape: [pri_nums, 5204]
    matches = truths[best_truth_idx]
    # 每个 pri框 对应的 gt框 的类别信息, Shape: [pri_nums], +1是因为将0作为背景框
    conf = labels[best_truth_idx] + 1
    # 把iou < threshold的框类别设置为背景类别, 即为0
    conf[best_truth_overlap < threshold] = 0
    # 保存匹配好的loc(需要编码, 编码后与net输出相比)和conf到loc_t和conf_t中
    loc = encode(matches, priors, variances)
    # idx: 一个batch中的图片id
    loc_t[idx] = loc
    conf_t[idx] = conf

    # 计算 预测框 与 真实框 的IoU --> RepGT
    # predicts为net的输出，解码后为框的实际表示
    predict_boxes = decode(predicts, priors, variances)
    overlaps = jaccard(
        truths,
        predict_boxes
    )
    # 观察IoU的内容
    # torch.count_nonzero(overlaps != -1)
    # 排除IOU最大元素
    # index为每个预测框对应的最大GT框的序号
    # scatter_相当于【根据坐标，把x值填入前置对象中】，即将IoU最大的置为-1
    index = torch.unsqueeze(best_truth_idx, 0)
    overlaps.scatter_(0, index, -1)
    # 排除非同类元素
    # conf - 1, 每个pri对应的gt框的类别，可以认为是预测框的类别
    # labels, gt框的类别信息
    # gt_labels = labels.repeat(conf.shape[0], 1).T
    # predict_labels = (conf - 1).repeat(labels.shape[0], 1)
    # overlaps = torch.where(gt_labels == predict_labels, overlaps, -torch.ones_like(overlaps))
    # 第二次匹配
    second_truth_overlap, second_truth_idx = overlaps.max(0, keepdim=True)
    # 排除第二次匹配内容
    overlaps.scatter_(0, second_truth_idx, -1)
    # 第三次匹配
    third_truth_overlap, third_truth_idx = overlaps.max(0, keepdim=True)
    third_truth_idx = torch.where((third_truth_overlap > 0) & (third_truth_overlap > (0.8 * second_truth_overlap)),
                                  third_truth_idx,
                                  second_truth_idx)
    # 选择对应的真实框坐标信息
    second_truth_idx.squeeze_(0)
    matches_G1 = truths[second_truth_idx]
    loc_g1[idx] = matches_G1
    third_truth_idx.squeeze_(0)
    matches_G2 = truths[third_truth_idx]
    loc_g2[idx] = matches_G2

    # 计算 预测框 与 预测框 的IoU --> RepBox
    # predict_boxes为解码后的位置表示
    # predict_objs为所有正样本预测框对应的真实框idx，注意是所有正样本
    predict_boxes = decode(predicts, priors, variances)
    predict_objs = torch.where(conf - 1 == -1, -torch.ones_like(best_truth_idx), best_truth_idx)
    repbox_nums = 0
    repbox_loss = torch.tensor(0.)
    # 增加遍历所有类别
    for class_idx in torch.unique(labels):
        # 遍历所有GT框
        for left in range(truths.shape[0]):
            # 如果该真实物体的类别不为当前遍历类别，则退出当前真实物体
            if labels[left] != class_idx:
                continue
            # 从对应真实物体为left的预测框中选择，随机采样
            left_predict_nums = torch.count_nonzero(predict_objs == left).item()
            if left_predict_nums == 0:
                continue
            random_idx = random.randint(0, left_predict_nums - 1)
            left_box_idx = torch.where(predict_objs == left)[0][random_idx]
            left_box = predict_boxes[left_box_idx].unsqueeze(0)
            # 计算Pi与其他所有预测框的IoU
            overlaps = jaccard(
                left_box,
                predict_boxes
            )
            # 从与left预测不同对象的预测框中选择
            for right in range(truths.shape[0]):
                # 如果该真实物体的类别不为当前遍历类别，则退出当前真实物体
                if labels[right] != class_idx:
                    continue
                # 如果左右为相同真实物体
                if left == right:
                    continue
                # 从对应真实物体为right的预测框中选择，随机采样
                right_predict_nums = torch.count_nonzero(predict_objs == right).item()
                if right_predict_nums == 0:
                    continue
                random_idx = random.randint(0, right_predict_nums - 1)
                right_box_idx = torch.where(predict_objs == right)[0][random_idx]
                x = overlaps[0][right_box_idx]
                if x.item() > 0:
                    repbox_nums += 1
                # 计算smoothln
                sigma = 0
                if x.item() <= sigma:
                    repbox_loss += -torch.log(1 - x + 1e-10)
                else:
                    repbox_loss += (x - sigma) / (1 - sigma) - torch.log(torch.tensor(1 - sigma))
    return repbox_loss, repbox_nums


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh + 1e-10) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_new(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# TODO: 修改算法测试
# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()  # shape[priors]
    if boxes.numel() == 0:  # shape[priors, 4]
        return keep

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # 按照得分排序
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]       # indices of the top-k largest vals

    # 循环进行NMS，count为循环的计数
    count = 0
    while idx.numel() > 0:  # idx中有多少个元素
        # 保存最大元素，并从idx中清除
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # 计算IoU = i / (area(a) + area(b) - i)
        # 加载剩余框的边界信息
        idx = torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        x1 = torch.autograd.Variable(x1, requires_grad=False)
        x1 = x1.data
        y1 = torch.autograd.Variable(y1, requires_grad=False)
        y1 = y1.data
        x2 = torch.autograd.Variable(x2, requires_grad=False)
        x2 = x2.data
        y2 = torch.autograd.Variable(y2, requires_grad=False)
        y2 = y2.data

        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        # 找剩余框与 m 的交集
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # 保证结果大于等于 0
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # 加载所有预测框的面积信息
        area = torch.autograd.Variable(area, requires_grad=False)
        area = area.data
        idx = torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou

        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
