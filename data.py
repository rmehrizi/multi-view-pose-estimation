# Rahil Mehrizi, Oct 2018
"""Loading Data from Disk"""

import os, sys
import numpy as np
import torch
import torch.utils.data as data

class PtsList(data.Dataset):
    def __init__( self, list_file, is_train=True):
        pts_list = [line.rstrip('\n') for line in open(list_file)]

        self.pts_list = pts_list
        self.is_train = is_train

    def __getitem__(self, index):
        token = self.pts_list[index].split(' ')

        # 2d poses
        pts_file = token[0]
        pts = np.loadtxt(pts_file)
        pts = pts.astype('float32')
        pts = torch.from_numpy(pts)

        # 3d poses and roots
        pts3d_file = token[1]
        pts3d = np.loadtxt(pts3d_file)
       
        R_file = token[2]  #rotation matrix
        R = np.loadtxt(R_file)

        pts3d_camera = np.matmul (np.linalg.inv(R) , np.transpose(pts3d))
        pts3d_camera = np.transpose(pts3d_camera)

        pts3d_camera = pts3d_camera.astype('float32')
        pts3d_camera = torch.from_numpy(pts3d_camera)

        return pts3d_camera, pts, pts_file

    def __len__(self):
        return len(self.pts_list)
