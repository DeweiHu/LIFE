#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 02:52:28 2020

@author: hud4
"""

import sys
sys.path.insert(0,'/home/hud4/Desktop/2020/RNN/')
import util
import os,pickle
import numpy as np
import matplotlib.pyplot as plt

'''
Re_Arrange reshape the volume from [nFrame*nBscan,H,W] -> [nFrame,nBscan,H,W]
'''
def Re_Arrange(volume):
    global nFrame
    n,H,W = volume.shape
    opt = np.zeros([nFrame,int(n/nFrame),H,W],dtype=np.float32)
    for i in range(n):
        idx = i % nFrame
        opt[idx,int(i/nFrame),:,:] = volume[i,:,:]
    return opt

global nFrame
nFrame = 5
dataroot = '/home/hud4/Desktop/2020/Human/'
var_pair = ()

for file in os.listdir(dataroot):
    if file.startswith('fovea') and file.endswith('.nii'):    
        print('volume: {}'.format(file))
        
        v_fovea = util.nii_loader(dataroot+file)
        v_fovea = Re_Arrange(np.transpose(v_fovea,(2,1,0)))
        _,nBscan,_,_ = v_fovea.shape
        
        for i in range(1,nBscan-1):
            slc = 10**v_fovea[:,i,50:562,24:-24]
            x = util.ImageRescale(v_fovea[0,i-1:i+2,50:562,24:-24],[0,255])
            var = util.ImageRescale(np.var(slc,axis=0),[0,1])
            var_pair = var_pair+((x,var),)
            
            # sample display
            if i % 300 == 0 and i != 0 :
                plt.figure(figsize=(12,6))
                plt.axis('off')
                plt.imshow(np.concatenate((x[1,:,:],util.ImageRescale(var,[0,255])),axis=1),cmap='gray')
                plt.show()
                
with open(dataroot+'var_pair.pickle','wb') as handle:
    pickle.dump(var_pair,handle)
    

