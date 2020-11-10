# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:54:06 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import pickle
import numpy as np
import matplotlib.pyplot as plt

dataroot = 'E:\\OCTA\\data\\R=3\\'

volume = ("fovea","fovea3","fovea5",
               "onh","onh2","onh5",
               "periph","periph3","periph4")
slc_range = ([80,120],[20,60],[20,60],
              [40,110],[35,105],[35,115],
              [30,50],[35,70],[30,55])
# cropping template
msk = [384,320]
    
def get_train_data(dataroot):
    train_data = ()
    for i in range(len(volume)):
        vol_x = util.nii_loader(dataroot+volume[i]+'.nii.gz')
        vol_y = util.nii_loader(dataroot+'sf('+volume[i]+').nii.gz')
        # extract vessel layers
        vol_x = vol_x[:,slc_range[i][0]:slc_range[i][1],:]
        vol_y = vol_y[:,slc_range[i][0]:slc_range[i][1],:]
        # size of en-face slices
        H, slc, W = vol_x.shape
        for j in range(slc):
            x = util.ImageRescale(vol_x[:,j,:],[0,255])
            y = util.ImageRescale(vol_y[:,j,:],[0,1])
            train_data = train_data + ((x[:msk[0],:msk[1]],y[:msk[0],:msk[1]]),
                                       (x[:msk[0],W-msk[1]:],y[:msk[0],W-msk[1]]),
                                       (x[H-msk[0]:,:msk[1]],y[H-msk[0]:,:msk[1]]),
                                       (x[H-msk[0]:,W-msk[1]:],y[H-msk[0]:,W-msk[1]]))
    return train_data
            
train_data = get_train_data(dataroot)

with open(dataroot+'train_data.pickle','wb') as func:
    pickle.dump(train_data,func)
    
    
    