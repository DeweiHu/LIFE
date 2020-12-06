# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:44:35 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np 

def cca(vol_cca, size_th=30):
	num_component = int(vol_cca.max())
	hist = np.histogram(vol_cca,num_component)
	size_vec = hist[0]

	cca_th = np.sum(np.uint8(size_vec>size_th))-1
	h,d,w = vol_cca.shape
	vol_opt = np.zeros(vol_cca.shape,dtype=np.int)

	for i in range(h):
		for j in range(d):
			for k in range(w):
				if (vol_cca[i,j,k]<=cca_th and vol_cca[i,j,k]>0):
					vol_opt[i,j,k] = 1

	return vol_opt


if __name__=="__main__":
    dataroot = 'E:\\OCTA\\binarize\\'
    vth = 15
    vol_cca = util.nii_loader(dataroot+'vol_cca.nii.gz')
    vol_opt = cca(vol_cca,vth)
    util.nii_saver(vol_opt,dataroot,'mask_TH={}.nii.gz'.format(vth))