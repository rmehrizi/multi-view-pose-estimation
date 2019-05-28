# Rahil Mehrizi, Oct 2018
"""Generating Training and Validation lists"""

import sys, os, random
import numpy as np
import PXIO
import os.path
import re

path_to_data = '.../multi-view-pose-estimation/dataset/'

"""Training"""
Subjects = [1, 5, 6, 7, 8]
lines = []

for s in Subjects:
    path = path_to_data + 'S' + str(s) + '/3D_positions_global/'
    actions = PXIO.ListSubfolderInFolder(path)
    for action in actions:  

            #2d pose
            pts_path1 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam1/'
            pts_path2 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam2/'
            pts_path3 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam3/'
            pts_path4 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam4/'

            #3d pose in global coordinates
            pts_path_3d = path + action + '/'

            #hip center in global coordinates
            root_path = path_to_data + 'S' + str(s) + '/Root/' + action + '/'

            #2d pose value
            V_path1 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam1/'
            V_path2 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam2/'
            V_path3 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam3/'
            V_path4 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam4/'

            #rotation matrix
            R_path = path_to_data + 'S' + str(s) + '/R/' 
            
            ds = '300W2/'
            pts_list = PXIO.ListFileInFolderRecursive(pts_path_3d, '.txt')

            for one_pts in pts_list:
                token = one_pts.split('/')
                pts_name = token[-1]
                fold = token[-2]

                pts_fpath1 = pts_path1 + pts_name
                pts_fpath2 = pts_path2 + pts_name
                pts_fpath3 = pts_path3 + pts_name
                pts_fpath4 = pts_path4 + pts_name

                pts_fpath_3d = pts_path_3d + pts_name

                root_fpath = root_path + pts_name

                V1 = V_path1 + pts_name
                V2 = V_path2 + pts_name
                V3 = V_path3 + pts_name
                V4 = V_path4 + pts_name

                R1 = R_path + '1.txt'
                R2 = R_path + '2.txt'
                R3 = R_path + '3.txt'
                R4 = R_path + '4.txt'

                if os.path.exists(pts_fpath1) and os.path.exists(pts_fpath2) and os.path.exists(pts_fpath3) and os.path.exists(pts_fpath4):
                              
                    line = '%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % (pts_fpath1, pts_fpath2, pts_fpath3, pts_fpath4, pts_fpath_3d, root_fpath, V1, V2, V3, V4, R1, R2, R3, R4 )
	            lines.append(line)

			  				 
print 'Total training images: %d' % len(lines)
random.shuffle(lines)
PXIO.WriteLineToFile('train_list_4view.txt', lines)


"""Validation"""
Subjects = [9, 11]
lines = []

for s in Subjects:
    path = path_to_data + 'S' + str(s) + '/3D_positions_global/'
    actions = PXIO.ListSubfolderInFolder(path)
    for action in actions:  

            #2d pose
            pts_path1 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam1/'
            pts_path2 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam2/'
            pts_path3 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam3/'
            pts_path4 = path_to_data + 'S' + str(s) + '/2D_positions_HG/' + action + '/Cam4/'

            #3d pose in global coordinates
            pts_path_3d = path + action + '/'

            #hip center in global coordinates
            root_path = path_to_data + 'S' + str(s) + '/Root/' + action + '/'

            #2d pose value
            V_path1 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam1/'
            V_path2 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam2/'
            V_path3 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam3/'
            V_path4 = path_to_data + 'S' + str(s) + '/value_HG/' + action + '/Cam4/'

            #rotation matrix
            R_path = path_to_data + 'S' + str(s) + '/R/' 
            
            ds = '300W2/'
            pts_list = PXIO.ListFileInFolderRecursive(pts_path_3d, '.txt')

            for one_pts in pts_list:
                token = one_pts.split('/')
                pts_name = token[-1]
                fold = token[-2]

                pts_fpath1 = pts_path1 + pts_name
                pts_fpath2 = pts_path2 + pts_name
                pts_fpath3 = pts_path3 + pts_name
                pts_fpath4 = pts_path4 + pts_name

                pts_fpath_3d = pts_path_3d + pts_name

                root_fpath = root_path + pts_name

                V1 = V_path1 + pts_name
                V2 = V_path2 + pts_name
                V3 = V_path3 + pts_name
                V4 = V_path4 + pts_name

                R1 = R_path + '1.txt'
                R2 = R_path + '2.txt'
                R3 = R_path + '3.txt'
                R4 = R_path + '4.txt'

                if os.path.exists(pts_fpath1) and os.path.exists(pts_fpath2) and os.path.exists(pts_fpath3) and os.path.exists(pts_fpath4):
                              
                    line = '%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % (pts_fpath1, pts_fpath2, pts_fpath3, pts_fpath4, pts_fpath_3d, root_fpath, V1, V2, V3, V4, R1, R2, R3, R4 )
	            lines.append(line)


#sorting based on frame name
convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
sorted_lines = sorted(lines, key = alphanum_key)
			  				 
print 'Total validation images: %d' % len(sorted_lines)
PXIO.WriteLineToFile('valid_list_4view.txt', sorted_lines)

#randomly select 10 samples to visualize
random.shuffle(sorted_lines)
del sorted_lines[6:]
print 'Total demo smaples: %d' % len(sorted_lines)
PXIO.WriteLineToFile('demo_list_4view.txt', sorted_lines)








