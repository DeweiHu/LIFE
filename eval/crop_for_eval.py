# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:32:42 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import matplotlib.pyplot as plt

dataroot = 'E:\\OCTA\\result\\'
saveroot = 'E:\\OCTA\\eval\\'

vol = util.nii_loader(dataroot+'orig5.nii.gz')
vol_seg = util.nii_loader(dataroot+'vol_seg5.nii.gz')
vol_base = util.nii_loader(dataroot+'binary5_base_2.nii.gz')
vol_binary = util.nii_loader('E:\\OCTA\\result5_TH=30.nii.gz')
vol_mseg = util.nii_loader('E:\\OCTA\\manualseg_fovea5.nii.gz')


#%%
h,slc,w = vol.shape
crop = np.zeros([100,slc,100],dtype=np.float32)
crop_seg = np.zeros([100,slc,100],dtype=np.float32)
crop_mseg = np.zeros([100,slc,100],dtype=np.float32)
crop_base = np.zeros([100,slc,100],dtype=np.float32)
crop_binary = np.zeros([100,slc,100],dtype=np.float32)

for i in range(slc):
    crop[:,i,:] = vol[220:320,i,180:280]
    crop_seg[:,i,:] = vol_seg[220:320,i,180:280]
    crop_mseg[:,i,:] = vol_mseg[220:320,i,180:280]
    crop_base[:,i,:] = vol_base[220:320,i,180:280]
    crop_binary[:,i,:] = vol_binary[220:320,i,180:280]
    
util.nii_saver(crop,saveroot,'crop.nii.gz')
util.nii_saver(crop_seg,saveroot,'crop_seg.nii.gz')
util.nii_saver(crop_mseg,saveroot,'crop_mseg.nii.gz')
util.nii_saver(crop_base,saveroot,'crop_base.nii.gz')
util.nii_saver(crop_binary,saveroot,'crop_binary.nii.gz')

#plt.figure(figsize=(12,12))
#plt.imshow(vol_seg[:,10,:],cmap='gray')
#plt.show()

