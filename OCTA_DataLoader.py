#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:30:46 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/20-summer/src/')
import util
import MotionCorrection as MC

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import io

'''
PickFrame separate the stacked repeated frames into single frame volume
FrameNum : number of repeated frame
idx : which frame to pick
'''
def PickFrame(volume,FrameNum,idx):
    dim = volume.shape
    opt = np.zeros([int(dim[0]/FrameNum),dim[1],dim[2]],dtype=np.float32)
    for i in range(dim[0]):
        if i % FrameNum == idx:
            opt[int(i/FrameNum),:,:] = volume[i,:,:]
    return opt

root = '/sdb/Data/Input/Human/'
FrameNum = 5

V_list = []
for file in os.listdir(root):
    if file.startswith('Retina2_Fovea') and file.endswith('.tif'):
        V_list.append(file)
V_list.sort()

#%%
raw = util.ImageRescale(io.imread(root+V_list[0]),[0,255])
V = np.zeros([FrameNum,500,1024,500],dtype=np.float32)

# re-arrange
for idx in range(FrameNum):
    V[idx,:,:,:] = PickFrame(raw,FrameNum,idx)

# Frame-registration
for slc in range(500):
    fix = np.ascontiguousarray(V[0,slc,:,:])
    for idx in range(FrameNum):
        mov = np.ascontiguousarray(V[idx,slc,:,:])
        reg = MC.MotionCorrect(fix,mov)
        V[idx,slc,:,:] = reg

# crop
#util.nii_saver(V[0,:,:,:],'/home/hud4/Desktop/','101_fovea.nii.gz')
V = V[:,:,150:150+512,:]

#%% speckle variance
slc = 158
im = V[:,slc,:,:]
#util.nii_saver(im,'/home/hud4/Desktop/','slc158.nii.gz')

var_map = np.mean(np.square(im-np.mean(im,axis=0)),axis=0)

plt.figure(figsize=(10,10))
plt.imshow(var_map)
plt.axis('off')
plt.show()
