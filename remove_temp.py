
import torch
from data import voc, VOCDetection, MEANS
from utils.augmentations import SSDAugmentation

if __name__ == '__main__':
    VOC_ROOT = 'D:\labs\object_detection_model\VOCdevkit'
    dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(voc['min_dim'], MEANS))
    for x in range(len(dataset)):
        dataset.pull_item(x)
