# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:21:24 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle,random
import numpy as np
import matplotlib.pyplot as plt

global dataroot, msk
# cropping template
msk = [320,320]
dataroot = 'E:\\OCTA\\data\\pre_processed\\'

volume = ("fovea","fovea3","fovea5")
slc_range = ([76,104],[13,43],[13,43])
    
def get_train_data(num):
    global msk
    train_data = ()
    for i in range(len(volume)):
        # pairing volumes
        vol_x = util.nii_loader(dataroot+'proj(orig)_'+volume[i]+'.nii.gz')
        vol_y = util.nii_loader(dataroot+'proj(var)_'+volume[i]+'.nii.gz')
        
        # extract vessel layers
        vol_x = vol_x[:,slc_range[i][0]:slc_range[i][1],:]
        vol_y = vol_y[:,slc_range[i][0]:slc_range[i][1],:]
        # size of en-face slices
        H, slc, W = vol_x.shape
        
        # iterate over the vessel layers
        for j in range(slc):
            x = util.ImageRescale(vol_x[:,j,:],[0,255])
            y = util.ImageRescale(vol_y[:,j,:],[0,255])
            
            # samples from single image
            for k in range(num):
                pseed = [random.randint(0,H-msk[0]),random.randint(0,W-msk[1])]
                im_x = x[pseed[0]:pseed[0]+msk[0],pseed[1]:pseed[1]+msk[1]]
                im_y = y[pseed[0]:pseed[0]+msk[0],pseed[1]:pseed[1]+msk[1]]
                train_data = train_data+((im_x,im_y),(np.fliplr(im_x),np.fliplr(im_y)),
                                         (np.flipud(im_x),np.flipud(im_y)),)      
    return train_data
            
train_data = get_train_data(10)

with open(dataroot+'proj(orig)_proj(var).pickle','wb') as func:
    pickle.dump(train_data,func)