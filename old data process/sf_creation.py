'''
import the whole volume, radius
output the self-fused volume
'''

import util,tool
import os
import numpy as np
import matplotlib.pyplot as plt

dataroot = '/home/dewei/Desktop/octa/data/'
temp = '/home/dewei/Desktop/octa/temp/'
volume = ["vol_octa"]
radius = 3


for i in range(len(volume)):
	vol = util.nii_loader(dataroot+volume[i]+'.nii.gz')
	vol = np.transpose(vol,[1,0,2])
	h,slc,w = vol.shape
	n_slc = slc - 2*radius

	# define the output volume
	vol_reg = np.zeros([h,n_slc,w],dtype=np.float32)
	vol_non_reg = np.zeros([h,n_slc,w],dtype=np.float32)

	for j in range(radius,slc-radius):
		stack = vol[:,j-radius:j+radius+1,:]
		stack_rg = tool.greedy(stack,temp)
		
		vol_non_reg[:,j-radius,:] = util.ImageRescale(tool.sf(stack,temp),[0,255])
		vol_reg[:,j-radius,:] = util.ImageRescale(tool.sf(stack_rg,temp),[0,255])

	# save the volume
	util.nii_saver(vol_non_reg,dataroot,'sf_'+volume[i]+'.nii.gz')
	util.nii_saver(vol_reg,dataroot,'sf_reg_'+volume[i]+'.nii.gz')
	print('volume {} self-fused.'.format(volume[i]))

print('Execution finished.')