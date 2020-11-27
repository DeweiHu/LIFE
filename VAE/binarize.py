# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:08:05 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\OCTA\\seg-VAE\\')
sys.path.insert(0,'E:\\tools\\')
import util
import cv2, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu
from medpy.filter.smoothing import anisotropic_diffusion

def Int8(im):
    im_uint = cv2.normalize(src=im,dst=0,alpha=0,beta=255,
                            norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    return im_uint

def ContrastEnhance(im):
    im = cv2.normalize(src=im,dst=0,alpha=0,beta=255,
                       norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    
    enhancer = ImageEnhance.Contrast(Image.fromarray(im))
    enhanced_im = enhancer.enhance(3.0)
    
    im_opt = cv2.normalize(src=np.array(enhanced_im),dst=0,alpha=0,beta=255,
                       norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    return im_opt

def binarize(vol,vol_seg,verbose):
    h,slc,w = vol.shape
    # define the output
    vol_base_1 = np.zeros(vol.shape,dtype=np.uint8) # otsu
    vol_base_2 = np.zeros(vol.shape,dtype=np.uint8) # diffuse+otsu
    vol_opt = np.zeros(vol.shape,dtype=np.uint8)    # seg+diffuse+otsu
    
    idx = random.randint(0,slc-1)
    for i in range(slc):
        # output 1
        im = Int8(vol[:,i,:])
        otsu_th_1 = threshold_otsu(im)
        vol_base_1[:,i,:] = np.uint8(im > otsu_th_1)*255
        
        # output 2
        diffuse = anisotropic_diffusion(im,niter=5,option=2).astype(np.float32)
        im_enhance = ContrastEnhance(diffuse)
        otsu_th_2 = threshold_otsu(im_enhance)
        vol_base_2[:,i,:] = np.uint8(im_enhance > otsu_th_2)*255
        
        # proposed
        im_seg = Int8(vol_seg[:,i,:])
        diffuse_seg = anisotropic_diffusion(im_seg,niter=5,option=2).astype(np.float32)
        im_enhance = ContrastEnhance(diffuse_seg)
        otsu_th_opt = threshold_otsu(im_enhance)
        vol_opt[:,i,:] = np.uint8(im_enhance > otsu_th_opt)*255
    
        if verbose == True and i == idx:
            plt.figure(figsize=(18,8))
            plt.axis('off')
            plt.title('base1 -- base2 -- proposed',fontsize=15)
            plt.imshow(np.concatenate((vol_base_1[:,i,:],vol_base_2[:,i,:],
                                       vol_opt[:,i,:]),axis=1),cmap='gray')
            plt.show()
            
    return vol_base_1, vol_base_2, vol_opt    

if __name__ == "__main__":        
    dataroot = "E:\\OCTA\\result\\"
    vol_seg = util.nii_loader(dataroot+"vol_seg3.nii.gz")
    vol = util.nii_loader(dataroot+"orig3.nii.gz")
    
    vol_base_1, vol_base_2, vol_opt = binarize(vol,vol_seg,True)
    
    util.nii_saver(vol_opt,dataroot,'binary3.nii.gz')
    util.nii_saver(vol_base_1,dataroot,'binary_base_1.nii.gz')
    util.nii_saver(vol_base_2,dataroot,'binary_base_2.nii.gz')