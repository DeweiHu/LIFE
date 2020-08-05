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
    opt = np.zeros([int(dim[2]/FrameNum),dim[0],dim[1]],dtype=np.float32)
    for i in range(dim[2]):
        if i % FrameNum == idx:
            opt[int(i/FrameNum),:,:] = volume[:,:,i]
    return opt

def Rotate(img):
    opt = np.fliplr(util.rot(img,'ccw'))
    return opt 

#%%
root = '/home/hud4/Desktop/20-summer/'
V = np.zeros([5,500,440,500],dtype=np.float32)
data = util.nii_loader(root+'Retina2 Fovea_SNR 101_reg.nii')

for i in range(5):
    V[i,:,:,:] = PickFrame(data,5,i)

V_var = np.var(V,axis=0)
util.nii_saver(Rotate(V_var),root,'var.nii.gz')

#%%
data = util.nii_loader(root+'PMFN_101{1}.nii.gz')
V_var = np.var(data[:,:,:440,:],axis=0)
util.nii_saver(Rotate(V_var),root,'var_dn.nii.gz')

#%%

FrameNum = 5

raw = util.nii_loader(root+'Retina2_Fovea_SNR_101_2.nii')
V = np.zeros([FrameNum,500,560,736],dtype=np.float32)

# re-arrange
for idx in range(FrameNum):
    V[idx,:,:,:] = PickFrame(raw,FrameNum,idx)

# Frame-registration
#for slc in range(500):
#    fix = np.ascontiguousarray(V[0,slc,:,:])
#    for idx in range(FrameNum):
#        mov = np.ascontiguousarray(V[idx,slc,:,:])
#        reg = MC.MotionCorrect(fix,mov)
#        V[idx,slc,:,:] = reg

util.nii_saver(util.ImageRescale(V[0,:,:,:],[0,255]),'/home/hud4/Desktop/','raw_slc0.nii.gz')
V_var = np.var(V,axis=0)
util.nii_saver(Rotate(V_var),root,'var_server.nii.gz')
