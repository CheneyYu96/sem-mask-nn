
import glob
import json
import os
import time
from os.path import abspath, dirname, isdir, isfile, join
import copy
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageFile
from torch.autograd import Variable
from torchvision import datasets
from torchvision import models as models
from torchvision.transforms import transforms

"""
    Default settings
"""
n_class    = 7

# TODO calculate mean of BGR
means = np.array([104.00698793, 116.66876762, 122.67891434])

"""
    global vars
"""
vals = {
    'use_gpu': 0,
    'batch_size': 1,
    'epochs': 30,
    'lr': 1e-4,
    'momentum': 0,
    'w_decay': 1e-5,
    'step_size': 10,
    'gamma': 0.5,
}

# dir path
ROOT_DIR = dirname(abspath(__file__))
MODEL_DIR = '{}/saved_model'.format(ROOT_DIR)
DATA_DIR = '{}/data'.format(ROOT_DIR)
PRED_DIR = '{}/pred'.format(ROOT_DIR)

now = lambda: time.time()
gap_time = lambda past_time : int((now() - past_time) * 1000)

def get_model_path(name, epoch):
    mkdir('{}/{}'.format(MODEL_DIR, name))
    return '{}/{}/{}'.format(MODEL_DIR, name, epoch)

def get_pred_name(infer, name):
    return '{}/{}/{}'.format(PRED_DIR, infer, name)

def mkdir(newdir):
    if type(newdir) is not str:
        newdir = str(newdir)
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            mkdir(head)
        if tail:
            os.mkdir(newdir)
