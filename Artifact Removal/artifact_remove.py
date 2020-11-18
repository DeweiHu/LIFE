# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:59:05 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import os
import matplotlib.pyplot as plt

def threshold(vec,th):
    if sum(vec) >= th:
        opt = 1
    else:
        opt = 0
    return opt

def Artifact_Remove(vol,th,maxIntensity):
    Ns, Nen, Nf = vol.shape
    vectors = np.mean(vol,axis=0)
    l,n = vectors.shape
    result = []
    
    for i in range(n):
        result.append(threshold(vectors[:,i],th))
    
    plt.figure(figsize=(8,6))
    plt.title('corrected slices',fontsize=15)
    plt.plot(result)
    plt.show()
    
    # define an output volume
    vol_opt = np.zeros([Ns,Nen,Nf],dtype=np.float32)
    
    # normal OCTA bscans are rescaled togather 
    n_normal = n-sum(result)
    stack_normal = np.zeros([Ns,Nen,n_normal],dtype=np.float32)
    
    idx = 0
    for i in range(Nf):
        if result[i] == 0:
            stack_normal[:,:,idx] = vol[:,:,i]
            idx += 1
    stack_normal = util.ImageRescale(stack_normal,[0,255])
    
    # abnormal OCTA bscans are rescaled independently
    idx = 0
    for i in range(Nf):
        if result[i] == 0: 
            vol_opt[:,:,i] = stack_normal[:,:,idx]
            idx += 1
        else:
            vol_opt[:,:,i] = util.ImageRescale(vol[:,:,i],[0,maxIntensity])
    return vol_opt

# [s-Bscan, en-face, f-Bscan]
dataroot = 'E:\\OCTA\\data\\R=3\\'
file = 'fovea5.nii.gz'
vol = util.nii_loader(dataroot+file)

vectors = np.mean(vol,axis=0)
plt.imshow(vectors,cmap='gray'),plt.show()
plt.plot(np.sum(vectors,axis=0))


#%%

th = 1.4e6
maxIntensity = 150

vol_opt = Artifact_Remove(vol,th,maxIntensity)    
util.nii_saver(vol_opt,dataroot,'AR_{}.nii.gz'.format(file[:file.find('.nii')]))
        
    
    




