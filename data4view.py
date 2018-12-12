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
        pts_file1 = token[0]
        pts_file2 = token[1]
        pts_file3 = token[2]
        pts_file4 = token[3]

        pts1 = np.loadtxt(pts_file1)
        pts2 = np.loadtxt(pts_file2)
        pts3 = np.loadtxt(pts_file3)
        pts4 = np.loadtxt(pts_file4)

        pts1 = pts1.astype('float32')
        pts1 = torch.from_numpy(pts1)
        pts2 = pts2.astype('float32')
        pts2 = torch.from_numpy(pts2)
        pts3 = pts3.astype('float32')
        pts3 = torch.from_numpy(pts3)
        pts4 = pts4.astype('float32')
        pts4 = torch.from_numpy(pts4)

        # 3d poses and roots
        pts3d_file = token[4]
        root_file = token[5]

        pts3d = np.loadtxt(pts3d_file)
        root = np.loadtxt(root_file)

        pts3d = pts3d.astype('float32')
        pts3d = torch.from_numpy(pts3d)

        root = root.astype('float32')
        root = torch.from_numpy(root)

        #2d pose values
        V_file1 = token[6]
        V_file2 = token[7]
        V_file3 = token[8]
        V_file4 = token[9]

        V1 = np.loadtxt(V_file1)
        V2 = np.loadtxt(V_file2)
        V3 = np.loadtxt(V_file3)
        V4 = np.loadtxt(V_file4)

        V1 = V1.astype('float32')
        V1 = torch.from_numpy(V1)
        V2 = V2.astype('float32')
        V2 = torch.from_numpy(V2)
        V3 = V3.astype('float32')
        V3 = torch.from_numpy(V3)
        V4 = V4.astype('float32')
        V4 = torch.from_numpy(V4)

        #rotation matrix
        R_file1 = token[10]
        R_file2 = token[11]
        R_file3 = token[12]
        R_file4 = token[13]

        R1 = np.loadtxt(R_file1)
        R2 = np.loadtxt(R_file2)
        R3 = np.loadtxt(R_file3)
        R4 = np.loadtxt(R_file4)

        R1 = R1.astype('float32')
        R1 = torch.from_numpy(R1)
        R2 = R2.astype('float32')
        R2 = torch.from_numpy(R2)
        R3 = R3.astype('float32')
        R3 = torch.from_numpy(R3)
        R4 = R4.astype('float32')
        R4 = torch.from_numpy(R4)

        return pts3d, root, pts1, pts2, pts3, pts4, R1, V1, R2, V2, R3, V3, R4, V4, pts3d_file

    def __len__(self):
        return len(self.pts_list)
