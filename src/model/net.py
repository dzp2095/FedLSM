# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import densenet
import copy
from functools import reduce

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

def set_module_by_name(module, access_string, target):
    names = access_string.split(sep='.')
    last = names[-1]
    names = names[:-1]
    module = reduce(getattr, names, module)
    setattr(module, last, target)

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, cfg):
        super(DenseNet121, self).__init__()
        num_classes = cfg['model']['num_classes']
        pretrained = cfg['model']['pretrained']
        drop_rate = cfg['model']['drop_rate']

        self.densenet121 = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate)
        
        num_ftrs = self.densenet121.classifier.in_features
        self._num_ftrs = num_ftrs
        # delete original classifer layer 
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            #nn.Sigmoid()
        )
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # feature shape 1024
        features = self.densenet121.features(x)
        
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        if self.drop_rate>0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.densenet121.classifier(out)
        return out

    @property
    def num_ftrs(self):
        return self._num_ftrs
        