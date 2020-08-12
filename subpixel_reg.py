# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:20:22 2020

@author: hudew
"""

import sys
sys.path.insert(0,'C:\\Users\\hudew\\OneDrive\\桌面\\Denoise\\')
import util

import numpy as np
import os
import matplotlib.pyplot as plt

from skimage import io
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

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

def Rotate(img):
    opt = np.fliplr(util.rot(img,'ccw'))
    return opt 

def SubpixelReg(fix,mov,acc):
    shifted, error, diffphase = register_translation(fix,mov,acc)    
    reg = shift(mov,shift=(shifted[0], shifted[1]), mode='constant')
    return reg

#%%
root = 'E:\\human\\' 
volumelist = []

for file in os.listdir(root):
    if file.startswith('Retina2_Fovea') and file.endswith('.tif'):
        volumelist.append(file)
volumelist.sort()

volume = volumelist[0]
raw = io.imread(root+volume)

Nf = 5
dim = raw.shape
x_range = [150,150+512]

#%% Register frames and Bscans
v = np.zeros(dim)
fix = raw[0,:,:]

for i in range(dim[0]):
    mov = raw[i,:,:]
    v[i,:,:] = SubpixelReg(fix,mov,100)

# [Nf,N_Bscan,H,W]
Vopt = np.zeros([Nf,int(dim[0]/Nf),int(x_range[1]-x_range[0]),dim[-1]],
              dtype=np.float32)

# re-arraneg and cropping
for idx in range(Nf):
    frame = PickFrame(v,Nf,idx)
    Vopt[idx,:,:,:] = frame[:,x_range[0]:x_range[1],:]
del v, frame

v_var = np.var(Vopt,axis=0)
util.nii_saver(Rotate(v_var),'E:\\OCTA\\','var2.nii.gz')

#%% Register only frames

# [Nf,N_Bscan,H,W]
v = np.zeros([Nf,int(dim[0]/Nf),int(x_range[1]-x_range[0]),dim[-1]],
              dtype=np.float32)

# re-arraneg and cropping
for idx in range(Nf):
    frame = PickFrame(raw,Nf,idx)
    v[idx,:,:,:] = frame[:,x_range[0]:x_range[1],:]
del raw, frame

# frame-registered volume
v_freg = np.zeros(v.shape,dtype=np.float32)

for i in range(500):
    fix = v[0,i,:,:]
    for j in range(Nf):
        mov = v[j,i,:,:]
        v_freg[j,i,:,:] = SubpixelReg(fix,mov,100)

v_var = np.var(v_freg,axis=0)
util.nii_saver(Rotate(v_var),'E:\\OCTA\\','var.nii.gz')        

#%% reference variance volume
root = 'E:\\OCTA\\'
Nf = 5

v_reg = util.nii_loader(root+'Retina2 Fovea_SNR 101_reg.nii')
v_reg = np.transpose(v_reg,[2,0,1])
v = np.zeros([Nf,500,440,500],dtype=np.float32)

for idx in range(Nf):
    v[idx,:,:,:] = PickFrame(v_reg,Nf,idx)
del v_reg

v_var = np.var(v,axis=0)

util.nii_saver(Rotate(v_var),root,'ref_var.nii.gz')