from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import args_test as args
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd


save_name = 'test1.txt'
save_gt_name = 'gt.txt'
save_pd_name = 'pred.txt'
test_dataset = [('2007', 'test')]
dataset_mean = (104, 117, 123)


if args['cuda'] and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args['save_folder']):
    os.mkdir(args['save_folder'])


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # 结果保存位置
    filename = save_folder + save_name
    num_images = len(testset)

    # 获取第 i 张图片
    for i in range(num_images):
        # 类似于从dataset中获取一张图片，只不过这里自己进行处理
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img_id, annotation = testset.pull_anno(i)
        img = testset.pull_image(i)
        img = transform(img)[0]
        # ---------------------------#
        img = img[:, :, (2, 1, 0)]
        # ---------------------------#
        x = torch.from_numpy(img).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        # 保存第 i 张图片的真实框
        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box))
                f.write(' || '+labelmap[box[4]]+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        # 对于每一类别
        for j in range(detections.size(1)):
            k = 0
            # 保存第 i 张图片中对于类别 j 的全部 k 个预测框
            while detections[0, j, k, 0] >= thresh:  # shape[1, classes, boxes, 5] conf + loc
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                # 图片预测的实际信息
                score = detections[0, j, k, 0]
                label_name = labelmap[j - 1]
                pt = (detections[0, j, k, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                k += 1


def test_net2(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    gt_filename = save_folder+save_gt_name
    pd_filename = save_folder+save_pd_name
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img_id, annotation = testset.pull_anno(i)
        img = testset.pull_image(i)
        img = transform(img)[0]
        # ---------------------------#
        img = img[:, :, (2, 1, 0)]
        # ---------------------------#
        x = torch.from_numpy(img).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(gt_filename, mode='a') as f:
            f.write(img_id+' ')
            for box in annotation:
                f.write(' '.join(str(b) for b in box)+' ')
            f.write('\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(pd_filename, mode='a') as f:
                        f.write(img_id+' ')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(pd_filename, mode='a') as f:
                    f.write(str(i-1) + ' ' + str(score) + ' ' +' '.join(str(c) for c in coords)+' ')
                j += 1
        with open(pd_filename, mode='a') as f:
            f.write('\n')


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1          # +1 background
    net = build_ssd('test', 300, num_classes)   # initialize SSD
    net.load_state_dict(torch.load(args['trained_model']))
    # 设置为eval模式，不执行dropout、BN等
    net.eval()
    print('Finished loading model!')

    # load data
    testset = VOCDetection(root=args['voc_root'],
                           image_sets=test_dataset,
                           # 由于test时，不确定数据集，所以具体处理输入图片的细节放在test_net中实现
                           transform=None)
    if args['cuda']:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_net(save_folder=args['save_folder'],
             net=net,
             cuda=args['cuda'],
             testset=testset,
             transform=BaseTransform(net.size, dataset_mean),
             thresh=args['visual_threshold'])


if __name__ == '__main__':
    test_voc()
