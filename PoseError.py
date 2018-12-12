# Rahil Mehrizi, Oct 2018
"""Calculating 3D Body Pose Error"""

import os, sys
import numpy as np
import torch

def Dist_Err(pred, target):
    assert(pred.size()==target.size())

    target = target.float()
    pred = pred.float()

    # distances between prediction and groundtruth coordinates
    dists = torch.zeros((pred.size(1), pred.size(0)))
    for i in range(pred.size(1)):
        for j in range(pred.size(0)):
            dists[i][j] = torch.dist(target[j][i], pred[j][i])


    avg_acc = dists.sum()/(dists.size(0)*dists.size(1))
    return avg_acc






 
