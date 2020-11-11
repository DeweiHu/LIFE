# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:54:06 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle,random
import numpy as np
import matplotlib.pyplot as plt

global dataroot, msk
dataroot = 'E:\\OCTA\\data\\R=3\\'
# cropping template
msk = [320,320]

volume = ("fovea","fovea3","fovea5",
          "periph","periph3","periph4")
slc_range = ([90,115],[25,55],[25,55],
              [35,48],[50,65],[40,45])
    
def get_train_data(num):
    global msk
    train_data = ()
    for i in range(len(volume)):
        vol_x = util.nii_loader(dataroot+volume[i]+'.nii.gz')
        vol_y = util.nii_loader(dataroot+'sf('+volume[i]+').nii.gz')
        # extract vessel layers
        vol_x = vol_x[:,slc_range[i][0]:slc_range[i][1],:]
        vol_y = vol_y[:,slc_range[i][0]:slc_range[i][1],:]
        # size of en-face slices
        H, slc, W = vol_x.shape
        
        # iterate over the vessel layers
        for j in range(slc):
            x = util.ImageRescale(vol_x[:,j,:],[0,255])
            y = util.ImageRescale(vol_y[:,j,:],[0,1])
            
            # samples from single image
            for k in range(num):
                pseed = [random.randint(0,H-msk[0]),random.randint(0,W-msk[1])]
                im_x = x[pseed[0]:pseed[0]+msk[0],pseed[1]:pseed[1]+msk[1]]
                im_y = y[pseed[0]:pseed[0]+msk[0],pseed[1]:pseed[1]+msk[1]]
                train_data = train_data+((im_x,im_y),(np.fliplr(im_x),np.fliplr(im_y)),
                                         (np.flipud(im_x),np.flipud(im_y)),)      
    return train_data
            
train_data = get_train_data(10)

with open(dataroot+'train_data.pickle','wb') as func:
    pickle.dump(train_data,func)
    
    
    