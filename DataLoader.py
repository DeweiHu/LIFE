#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:01:58 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/20-summer/src/')
import util

import os, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage import io
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

def SubpixelReg(fix,mov,acc):
    shifted, error, diffphase = register_translation(fix,mov,acc)    
    reg = shift(mov,shift=(shifted[0], shifted[1]), mode='constant')
    return reg

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

root = '/home/hud4/Desktop/20-summer/'
root_tif = '/sdb/Data/Input/Human/'

#%% registered example
f0 = 'Retina2 Fovea_SNR 101_reg.nii'
data = np.transpose(util.nii_loader(root+f0),[2,1,0])
dim = data.shape

# re-arrange
Nf = 5
v0 = np.zeros([Nf,int(dim[0]/Nf),dim[1],dim[2]],dtype=np.float32)

for i in range(Nf):
    v0[i,:,:,:] = PickFrame(data,Nf,i)
del data
v0 = np.transpose(v0,[0,1,3,2])

v0_var = np.var(v0,axis=0)

#%% tiff volume
f1 = 'Retina2_Fovea_SNR_101.tif'
data = io.imread(root_tif+f1)
data = data[:,100:600,:]
dim = data.shape

# re-arrange
v1 = np.zeros([Nf,int(dim[0]/Nf),dim[1],dim[2]],dtype=np.float32)

for i in range(Nf):
    v1[i,:,:,:] = PickFrame(data,Nf,i)
del data

v1_var = np.var(v1,axis=0)

#%% "corrected zero-padding" 
f2 = 'Retina2_Fovea_SNR_101_2.nii'
data = util.nii_loader(root+f2)
data = np.transpose(data,[2,1,0])
dim = data.shape

# re-arrange
Nf = 5
v2 = np.zeros([Nf,int(dim[0]/Nf),dim[1],dim[2]],dtype=np.float32)

for i in range(Nf):
    v2[i,:,:,:] = PickFrame(data,Nf,i)
del data

v2_var = np.var(v2,axis=0)

#%%
slc = 411
im_2 = v2[0,slc,320:550,5:505]
im_0 = v0[0,slc,180:410,:] 

plt.figure(figsize=(15,15))
plt.subplot(2,1,1),plt.imshow(im_0,cmap='gray')
plt.subplot(2,1,2),plt.imshow(im_2,cmap='gray')

#%%
import numpy.matlib

util.nii_saver(Rotate(v0[:,slc,180:410,:]),root,'v0_slc{}.nii'.format(slc))
util.nii_saver(Rotate(v2[:,slc,320:550,5:505]),root,'v2_slc{}.nii'.format(slc))

var_2 = v2_var[slc,320:550,5:505]
#var_0 = v0_var[slc,180:410,:]

plt.figure(figsize=(15,15))
plt.subplot(2,1,1),plt.imshow(var_0,cmap='gray')
plt.subplot(2,1,2),plt.imshow(var_2,cmap='gray')

v2_var = np.zeros([Nf,230,500],dtype=np.float32)
for i in range(Nf):
    v2_var[i,:,:] = var_2
util.nii_saver(Rotate(v2_var),root,'var2_slc{}.nii'.format(slc))

#%%
util.nii_saver(Rotate(np.transpose(v0[0,:,:,:],[0,2,1])),root,'v0.nii.gz')
util.nii_saver(Rotate(v1[0,:,:,:]),root,'v1.nii.gz')
util.nii_saver(Rotate(v2[0,:,160:600,:]),root,'v2.nii.gz')
