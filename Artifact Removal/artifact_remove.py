# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:03:59 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import os
import numpy as np
import matplotlib.pyplot as plt

def threshold(vec,th):
    if sum(vec) >= th:
        opt = 1
    else:
        opt = 0
    return opt


def artifact_removal(vol,crop_range):
    # classification
    vectors = np.mean(vol,axis=0)
    v = np.sum(vectors,axis=0)
    mu = np.mean(v)
    std = np.std(v)
    th = mu + std
    v_class = v > th
    
    # crop
    vol_crop = vol[:,crop_range[0]:crop_range[1],:]
    vol_opt = np.zeros(vol_crop.shape,dtype=np.float32)
    
    for i in range(len(v_class)):
        if v_class[i]:
            im_abn = vol_crop[:,:,i]
            idx = i
            # jump out of while loop when get a normal slice
            while v_class[idx]:
                idx -= 1
            im_n = vol_crop[:,:,idx]
            
            # histogram matching
            im_hm = util.hist_match(im_abn,im_n)
            vol_opt[:,:,i] = im_hm
            
        else:
            vol_opt[:,:,i] = vol_crop[:,:,i]
    
    return vol_crop, vol_opt

if __name__=="__main__":
    dataroot = 'E:\\OCTA\\data\\R=3\\'
    saveroot = 'E:\\OCTA\\data\\AR_result\\'
    volume = ("fovea","fovea3","fovea5")
    slc_range = ([76,104],[13,43],[13,43])
    
    for file in os.listdir(dataroot):
        for i in range(len(volume)):
            vol = util.nii_loader(dataroot+volume[i]+'.nii.gz')
            vol_crop,vol_opt = artifact_removal(vol,slc_range[i])
            
            util.nii_saver(vol_crop,saveroot,volume[i]+'_crop.nii.gz')
            util.nii_saver(vol_opt,saveroot,volume[i]+'_AR.nii.gz')
      
  
            
        
    
    