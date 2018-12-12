# Xi Peng, Jun 16 2016
import os, sys, shutil, random
import numpy as np

def ListSubfolderInFolder(path):
    return [f for f in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path,f))]

def ListFileInFolder(path,format):
    list = []
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(format):
                list.append(root+file)
    return list

def ListFileInFolderRecursive(path,format):
    list = [];
    for root, dirs, files in os.walk(path):
       ## for fold in dirs:
        files = os.listdir(root)
        for file in sorted(files):
            if file.endswith(format):
                list.append(root+file)
    return list

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteLineToFile(path,lines):
    with open(path, 'w') as fd:
        for line in lines:
            fd.write(line + '\n')

def WriteLineToFileShuffle(path,lines):
    random.shuffle(lines)
    with open(path, 'w') as fd:
        for line in lines:
            fd.write(line + '\n')

def ReadFloatFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            line = line.rstrip('\n').split(' ')
            line2 = [float(line[i]) for i in range(len(line))]
            lines.append(line2)
    lines = np.array(lines)
    return lines

def ReadAnnotMPII(path):
    annot = {}
    with open(path, 'r') as fd:

        annot['imgName'] = next(fd).rstrip('\n')        
        annot['center'] = [int(x) for x in next(fd).split()]
        annot['scale'] = float(next(fd).rstrip('\n'))
        annot['pts'] = []
        annot['vis'] = []

        for line in fd:
            x, y, isVis = [int(float(x)) for x in line.split()]
            annot['pts'].append((x,y))
            annot['vis'].append(isVis)
    return annot

def ReadAnnotInfo(path):
    annot = {}
    with open(path, 'r') as fd:
        annot['center'] = [int(x) for x in next(fd).split()]
        annot['scale'] = float(next(fd).rstrip('\n'))
        annot['pts'] = []
        annot['vis'] = []
        for line in fd:
            x, y, isVis = [int(float(x)) for x in line.split()]
            annot['pts'].append((x,y))
            annot['vis'].append(isVis)
    return annot


def SaveAugPtsToFile(pts_spath, pts_aug_objs):
    pts_aug = np.zeros((len(pts_aug_objs), 2))
    for i in range(len(pts_aug_objs)):
        pts_aug[i, 0] = pts_aug_objs[i].x
        pts_aug[i, 1] = pts_aug_objs[i].y
    np.savetxt(pts_spath, pts_aug, fmt='%.1f')

def WriteFloatToFile(path,lines):
    with open(path,'w') as fd:
        print lines.shape
        print lines.ndim 

def DeleteThenCreateFolder(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def CreateFolderIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__=='__main__':
    print 'Python IO Lib by Xi Peng'
