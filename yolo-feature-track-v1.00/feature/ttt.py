import argparse
import os

import cv2
import numpy
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torchvision.models import resnet18

from dataset import create_datasets, Dataset
from main import get_dataset_dir
from models import Resnet18FaceModel
import matplotlib.pyplot as plt


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = Conv2d(3,3,3)
        self.bn1 = BatchNorm2d(3)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

def fun0():
    training_losses = {
        'center': [], 'cross_entropy': [],
        'together': [], 'top3acc': [], 'top1acc': []}

    with open(r"plot/test.txt", "w") as f:
        f.write(training_losses)

def fun1(losses):

    fig = plt.figure("train")
    plt.xlabel('epochs')
    plt.ylabel('loss')

    for loss in losses:
        plt.plot(loss[0], loss[1])
    plt.show()
    # fig.savefig(path, dpi=fig.dpi)

def fun2():
    return (
        [[1,2,3,4,5,6,7],[2,3,4,1,2,3,7]],
    )

def fun3():
    model = resnet18()
    print(type(model))
def fun4():
    train_info = r"plot/train," + "resnet18" + ",lamda=" + "0.03" + r".txt"
    print(1 if True else 2)

if __name__ == '__main__':
    fun4()