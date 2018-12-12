# Xi Peng, Feb 2017

import os, sys
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch

class TrainHistory():
    """store statuses from the 1st to current epoch"""
    def __init__(self):
        self.epoch = []
        self.lr = []
        self.loss = []
        self.pckh = []
        self.best_pckh = 0.
        self.is_best = True

    def update(self, epoch, lr, loss, pckh):
        self.epoch.append(epoch)
        self.lr.append(lr)
        self.loss.append(loss)
        self.pckh.append(pckh)

        self.is_best = pckh['val_pckh'] > self.best_pckh
        self.best_pckh = max(pckh['val_pckh'], self.best_pckh)

    def state_dict(self):
        dest = OrderedDict()
        dest['epoch'] = self.epoch
        dest['lr'] = self.lr
        dest['loss'] = self.loss
        dest['pckh'] = self.pckh
        dest['best_pckh'] = self.best_pckh
        dest['is_best'] = self.is_best
        return dest

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.loss = state_dict['loss']
        self.pckh = state_dict['pckh']
        self.best_pckh = state_dict['best_pckh']
        self.is_best = state_dict['is_best']

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

