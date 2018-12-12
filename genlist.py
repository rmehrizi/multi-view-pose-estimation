# Rahil Mehrizi, Oct 2018
"""Generating Training and Validation lists"""

import sys, os, random
import numpy as np
import PXIO
import os.path
import re

path_to_data = '/home/rahil/PoseEstimator/dataset/'

"""Training"""
Subjects = [1, 5, 6, 7, 8]
lines = []

for c in range(1,5):
    for s in Subjects:
        path = path_to_data + 'S' + str(s) + '/3D_positions_global/'
        actions = PXIO.ListSubfolderInFolder(path)
        for action in actions:  

            #2d pose
            pts_path = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam'+ str(c) +'/'

            #3d pose in global coordinates
            pts_path_3d = path + action + '/'

            #rotation matrix
            R_path = path_to_data + 'S' + str(s) + '/R/' 
            
            ds = '300W2/'
            pts_list = PXIO.ListFileInFolderRecursive(pts_path_3d, '.txt')

            for one_pts in pts_list:
                token = one_pts.split('/')
                pts_name = token[-1]
                fold = token[-2]

                pts_fpath = pts_path + pts_name

                pts_fpath_3d = pts_path_3d + pts_name

                R = R_path + str(c) + '.txt'


                if os.path.exists(pts_fpath):
                              
                    line = '%s %s %s' % (pts_fpath, pts_fpath_3d, R)
	            lines.append(line)
			  				 
print 'Total training images: %d' % len(lines)
random.shuffle(lines)
PXIO.WriteLineToFile('train_list.txt', lines)


"""Validation"""
Subjects = [9, 11]
lines = []

for c in range(1,5):
    for s in Subjects:
        path = path_to_data + 'S' + str(s) + '/3D_positions_global/'
        actions = PXIO.ListSubfolderInFolder(path)
        for action in actions:  

            #2d pose
            pts_path = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam'+ str(c) +'/'

            #3d pose in global coordinates
            pts_path_3d = path + action + '/'

            #rotation matrix
            R_path = path_to_data + 'S' + str(s) + '/R/' 
            
            ds = '300W2/'
            pts_list = PXIO.ListFileInFolderRecursive(pts_path_3d, '.txt')

            for one_pts in pts_list:
                token = one_pts.split('/')
                pts_name = token[-1]
                fold = token[-2]

                pts_fpath = pts_path + pts_name

                pts_fpath_3d = pts_path_3d + pts_name

                R = R_path + str(c) + '.txt'


                if os.path.exists(pts_fpath):
                              
                    line = '%s %s %s' % (pts_fpath, pts_fpath_3d, R)
	            lines.append(line)

#sorting based on frame name
convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
sorted_lines = sorted(lines, key = alphanum_key)
			  				 
print 'Total validation images: %d' % len(sorted_lines)
PXIO.WriteLineToFile('valid_list.txt', sorted_lines)










