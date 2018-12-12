# Rahil Mehrizi, Oct 2018
"""Visualizing 2D and 3D Poses"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Plot2d(pose, ax):
    ax.plot(pose[0,[7,8,9],0] , -pose[0,[7,8,9],1] , color='b')
    ax.plot(pose[0,[7,1],0] , -pose[0,[7,1],1] , color='b')
    ax.plot(pose[0,[7,4],0] , -pose[0,[7,4],1] , color='b')
    ax.plot(pose[0,[8,10],0] , -pose[0,[8,10],1] , color='b')
    ax.plot(pose[0,[8,13],0] , -pose[0,[8,13],1] , color='b')
    ax.plot(pose[0,[1,2,3],0] , -pose[0,[1,2,3],1] , color='r')
    ax.plot(pose[0,[4,5,6],0] , -pose[0,[4,5,6],1] , color='g')
    ax.plot(pose[0,[10,11,12],0] , -pose[0,[10,11,12],1] , color='r')
    ax.plot(pose[0,[13,14,15],0] , -pose[0,[13,14,15],1] , color='g')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    
   # return plt 

def Plot3d(pose, ax, colors):
    ax.plot(pose[0,[6,7,8,9],0] , pose[0,[6,7,8,9],1] , pose[0,[6,7,8,9],2] , color = colors[0])
    ax.plot(pose[0,[7,10],0] , pose[0,[7,10],1] , pose[0,[7,10],2] , color = colors[0])
    ax.plot(pose[0,[7,13],0] , pose[0,[7,13],1] , pose[0,[7,13],2], color = colors[0])
    ax.plot(pose[0,[6,0],0] , pose[0,[6,0],1] , pose[0,[6,0],2] , color = colors[0])
    ax.plot(pose[0,[6,3],0] , pose[0,[6,3],1] , pose[0,[6,3],2] , color = colors[0])
    ax.plot(pose[0,[0,1,2],0] , pose[0,[0,1,2],1] , pose[0,[0,1,2],2] , color = colors[1])
    ax.plot(pose[0,[3,4,5],0] , pose[0,[3,4,5],1] , pose[0,[3,4,5],2] , color = colors[2])
    ax.plot(pose[0,[10,11,12],0] , pose[0,[10,11,12],1] , pose[0,[10,11,12],2] , color = colors[1])
    ax.plot(pose[0,[13,14,15],0] , pose[0,[13,14,15],1] , pose[0,[13,14,15],2] , color = colors[2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    ax.axis('equal')
    ax.w_xaxis.set_pane_color((1,1,1,0))
    ax.w_yaxis.set_pane_color((1,1,1,0))
    ax.w_xaxis.line.set_color((1,1,1,0))
    ax.w_yaxis.line.set_color((1,1,1,0))
    ax.w_zaxis.line.set_color((1,1,1,0))

    ax.view_init(elev=20, azim=-62)




