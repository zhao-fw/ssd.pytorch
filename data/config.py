# config.py
import os.path as osp


HOME = osp.abspath("D:/workspace/")
# HOME = osp.expanduser("~")
# HOME = osp.abspath(".")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# TODO:图片均值
MEANS = (104, 117, 123)

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    # 'max_iter': 500,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # 长宽比
    'variance': [0.1, 0.2],  # 方差
    'clip': True,
    'name': 'VOC',
}

args = {
    'dataset': 'VOC',
    'dataset_root': VOC_ROOT,
    'basenet': 'vgg16_reducedfc.pth',
    'batch_size': 16,
    'resume': None,  # Checkpoint state_dict file to resume training from
    'start_iter': 0,  # Resume training at this iter
    'num_workers': 4,  # Number of workers used in dataloading
    'cuda': True,
    'device_ids': [0],  # DataParallel参数
    'lr': 1e-3,  # learning rate
    'momentum': 0.9,  # Momentum value for optim
    'weight_decay': 5e-4,  # Weight decay for SGD
    'gamma': 0.1,  # Gamma update for SGD
    'visdom': False,
    'save_folder': 'weights/',  # Directory for saving checkpoint models
}

args_eval = {
    'trained_model': 'weights/ssd300_mAP_77.43_v2.pth',  # 评估模型
    'save_folder': 'eval/',  # File path to save results
    'confidence_threshold': 0.01,  # Detection confidence threshold
    'top_k': 5,  # Further restrict the number of predictions to parse
    'cuda': True,
    'voc_root': VOC_ROOT,
    'cleanup': True  # Cleanup and remove results files following eval
}

args_test = {
    'trained_model': 'weights/ssd_300_VOC0712.pth',  # Trained state_dict file path to open
    'save_folder': 'eval/',  # Dir to save results
    'visual_threshold': 0.6,  # Final confidence threshold
    'cuda': True,
    'voc_root': VOC_ROOT,  # Location of VOC root directory
    'f': None  # Dummy arg so we can load in Jupyter Notebooks
}
