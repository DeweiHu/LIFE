# -*- coding: utf-8 -*-
"""
The possible pseudo-modal
1. local-projection of orig
2. self-fusion of orig
3. var 
4. local projection of var
5. self-fusion of var
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import os
import matplotlib.pyplot as plt

'''
vol: [s-bscan, en-face, f-bscan]
 r : radius that within which neighbors will be considered for variance/mean
start : starting slice
end : ending slice
request : 'var','mean'
'''
def loc_Proc(vol,r,start,end,request):
    h,slc,w = vol.shape
    if not (start>=r and end<=slc-r):
        raise ValueError('Unfit range selected.')
    vol_opt = np.zeros([h,end-start,w], dtype=np.float32)
    
    for i in range(start,end):
        if request == 'var':
            vol_opt[:,i-start,:] = np.var(vol[:,i-r:i+r+1,:],axis=1)
        elif request == 'mean':
            vol_opt[:,i-start,:] = np.mean(vol[:,i-r:i+r+1,:],axis=1)
        else:
            raise ValueError('request invalid.')
    return vol_opt

def cutter(vol1,vol2):
    _,n1,_ = vol1.shape
    _,n2,_ = vol2.shape
    if n1 > n2:
        r = int((n1-n2)/2)
        opt1 = vol1[:,r:n1-r,:]
        opt2 = vol2
    else:
        r = int((n2-n1)/2)
        opt1 = vol1
        opt2 = vol2[:,r:n2-r,:]
    return opt1,opt2
    
dataroot = 'E:\\OCTA\\data\\R=3\\'
saveroot = 'E:\\OCTA\\data\\pre_processed\\'
r_var = 3
r_proj = 5

for file in os.listdir(dataroot):
    if file.startswith('AR') and file.endswith('.nii.gz'):
        vol = util.nii_loader(dataroot+file)
        h,slc,w = vol.shape
        # local_proj{orig}
        orig_proj = loc_Proc(vol,3,3,slc-3,'mean')
        # var
        var = loc_Proc(vol,r_var,r_var,slc-r_var,'var')
        # local_proj{var}
        _,slc,_ = var.shape
        var_proj = loc_Proc(var,r_proj,r_proj,slc-r_proj,'mean')
        
        vol_vp,vol_v = cutter(var_proj,var)
        _,vol_op = cutter(var_proj,orig_proj)
        _,vol_orig = cutter(var_proj,vol)
        
        idx = file.find('_')
        util.nii_saver(vol_orig,saveroot,'orig_{}'.format(file[idx:]))
        util.nii_saver(util.ImageRescale(vol_op,[0,255]),
                       saveroot,'proj(orig)_{}'.format(file[idx:]))
        util.nii_saver(util.ImageRescale(vol_v,[0,255]),
                       saveroot,'var_{}'.format(file[idx:]))
        util.nii_saver(util.ImageRescale(vol_vp,[0,255]),
                       saveroot,'proj(var)_{}'.format(file[idx:]))

