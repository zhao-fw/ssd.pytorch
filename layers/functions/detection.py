import torch
import torch.nn as nn
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors, num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1, num_priors, 4]
        """
        num_batch = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num_batch, self.num_classes, self.top_k, 5)  # 初始化输出
        conf_preds = conf_data.view(num_batch, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num_batch):
            # 解码loc
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # 拷贝每个batch内的conf，用于nms
            conf_scores = conf_preds[i].clone()

            # 遍历每一个类别（上面conf_data交换维度）
            for cl in range(1, self.num_classes):
                # 筛选掉 conf < conf_thresh 的conf
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                # 筛选掉 conf < conf_thresh 的loc
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # 修改阈值为1，观察未使用NMS的效果
                # ids, count = nms(boxes, scores, 1, self.top_k)
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        # TODO: 这是？？？
        flt = output.contiguous().view(num_batch, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
