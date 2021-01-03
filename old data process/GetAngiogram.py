# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 21:01:44 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np

global nFrame
nFrame = 5

def GetAngiogram(vol_raw):
    global nFrame
    h,w,n_slc = vol_raw.shape
    vol_sep = np.zeros([nFrame,h,w,int(n_slc/nFrame)],dtype=np.float32)
    
    for i in range(n_slc):
        frame = int(i % nFrame)
        slc = int(np.floor(i/nFrame))
        vol_sep[frame,:,:,slc] = vol_raw[:,:,i]
    
    vol_opt = np.var(vol_sep,axis=0)
    return vol_opt

if __name__ == '__main__':        
    dataroot = 'C:\\Users\\hudew\\Downloads\\'
    vol_raw = np.float32(util.nii_loader(dataroot+'fovea101_reg.nii'))
    vol_octa = GetAngiogram(vol_raw)
    d_range = [220,280]     
    util.nii_saver(vol_octa[d_range[0]:d_range[1],:,:],'E:\\','vol_octa.nii.gz')
