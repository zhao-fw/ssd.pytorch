from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np


now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
train_dataset = [('2007', 'trainval'), ('2012', 'trainval')]
dataset_mean = (104, 117, 123)


if torch.cuda.is_available():
    if args['cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args['cuda']:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if not os.path.exists(args['save_folder']):
    os.mkdir(args['save_folder'])


def train():
    # 数据集
    cfg = voc
    dataset = VOCDetection(root=args['dataset_root'],
                           image_sets=train_dataset,
                           transform=SSDAugmentation(cfg['min_dim'], dataset_mean))

    # 网络
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # 参数
    if args['visdom']:
        import visdom
        global viz
        viz = visdom.Visdom()
    if args['cuda']:
        net = torch.nn.DataParallel(ssd_net, device_ids=args['device_ids'])
        cudnn.benchmark = True
    if args['resume']:
        print('Resuming training, loading {}...'.format(args['resume']))
        ssd_net.load_weights(args['resume'])
    else:
        vgg_weights = torch.load(args['save_folder'] + args['basenet'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
    if args['cuda']:
        net = net.cuda()
    if not args['resume']:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # 优化器，更新参数
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'],
                          weight_decay=args['weight_decay'])
    # TODO:多尺度损失函数，部分参数
    criterion = MultiBoxLoss(num_classes=cfg['num_classes'],
                             overlap_thresh=0.5,
                             prior_for_matching=True,
                             bkg_label=0,
                             neg_mining=True,
                             neg_pos=3,
                             neg_overlap=0.5,
                             encode_target=False,
                             use_gpu=args['cuda']
                             )

    # 设置为train模式，dropout层、BN层等
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    repul_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args['batch_size']
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args['visdom']:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Repul Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset,
                                  args['batch_size'],
                                  num_workers=args['num_workers'],
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True,
                                  generator=torch.Generator(device='cuda'))
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args['start_iter'], cfg['max_iter']):
        # 可视化: epoch
        if args['visdom'] and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, loc_loss, repul_loss, conf_loss, epoch_plot, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            repul_loss = 0
            conf_loss = 0

        # 在某些时间段修改学习率
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args['gamma'], step_index)

        # 获取一次batch的数据
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        # 是否GPU
        if args['cuda']:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_l_repul, loss_c = criterion(out, targets)
        loss = loss_l + loss_c + loss_l_repul
        loss.backward()
        optimizer.step()

        # 时间
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        repul_loss += loss_l_repul.item()

        # 打印输出
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f' % (loss.item()) +
                  ' || conf_loss: %.4f ' % (loss_c.item()) +
                  ' || smoothl1 loss: %.4f ' % (loss_l.item()) +
                  ' || repul loss: %.4f ||' % (loss_l_repul.item()), end=' ')
        # 可视化: iteration
        if args['visdom']:
            update_vis_plot(iteration, loss_l.item(), loss_l_repul.item(), loss_c.item(), iter_plot, 'append')
            # initialize epoch plot on first iteration
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 4)).cpu(),
                    Y=torch.Tensor([loss_l.item(), loss_l_repul.item(), loss_c.item(),
                                    loss_l.item() + loss_c.item()]).unsqueeze(0).cpu(),
                    win=epoch_plot,
                    update=True
                )

        # 暂存模型
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' + now_time +
                       repr(iteration) + '.pth')
    # finally save
    torch.save(ssd_net.state_dict(),
               args['save_folder'] + '' + args['dataset'] + now_time + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args['lr'] * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 4)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, repul, conf, window, update_type, epoch_size=1):
    viz.line(
        X=torch.ones((1, 4)).cpu() * iteration,
        Y=torch.Tensor([loc, repul, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window,
        update=update_type
    )


if __name__ == '__main__':
    train()
