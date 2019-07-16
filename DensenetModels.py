import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision
import se_densenet


class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):

        super(ResNet50, self).__init__()

        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features

        self.resnet50.fc = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        x = self.resnet50(x)
        return x
import se_resnet

class SE_ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):

        super(SE_ResNet50, self).__init__()

        self.se_resnet50 = se_resnet.se_resnet50(num_classes = classCount,pretrained=isTrained)


    def forward(self, x):
        x = self.se_resnet50(x)
        return x

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x
class SE_DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(SE_DenseNet121, self).__init__()

        self.densenet121 = se_densenet.densenet121(pretrained=isTrained)

        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet169, self).__init__()

        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)

        kernelCount = self.densenet169.classifier.in_features

        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward (self, x):
        x = self.densenet169(x)
        return x

class DenseNet201(nn.Module):

    def __init__ (self, classCount, isTrained):

        super(DenseNet201, self).__init__()

        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)

        kernelCount = self.densenet201.classifier.in_features

        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward (self, x):
        x = self.densenet201(x)
        return x
