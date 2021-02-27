import sys
sys.path.insert(0,'E:\\tools\\')
import util
import cv2, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.filters import threshold_otsu, threshold_local
from medpy.filter.smoothing import anisotropic_diffusion
from sklearn.cluster import KMeans

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

def Kmeans(im):
    h,w = im.shape    
    kmeans = KMeans(n_clusters=2)
    diffuse = anisotropic_diffusion(im,niter=5,option=2).astype(np.float32)
    im_enhance = ContrastEnhance(diffuse)
    vectorized = im_enhance.reshape((-1,1))
    kmeans.fit(vectorized)
    y_kmeans = kmeans.predict(vectorized)
    y = y_kmeans.reshape((h,w))
    return y

def binarize(vol,vol_seg,verbose):
    h,slc,w = vol.shape
    # define the output
    vol_base_1 = np.zeros(vol.shape,dtype=np.uint8) # diffuse+kmeans
    vol_base_2 = np.zeros(vol.shape,dtype=np.uint8) # diffuse+otsu
    vol_opt = np.zeros(vol.shape,dtype=np.uint8)    # seg+diffuse+otsu
    

    idx = random.randint(0,slc-1)
    for i in range(slc):
        # output 1
        im = Int8(vol[:,i,:])
        vol_base_1[:,i,:] = np.uint8(Kmeans(im))*255
        
        # output 2
        diffuse = anisotropic_diffusion(im,niter=10,option=2).astype(np.float32)
        im_enhance = ContrastEnhance(diffuse)
        otsu_th = threshold_otsu(im_enhance)
        vol_base_2[:,i,:] = np.uint8(im_enhance > otsu_th)*255
        
        # proposed
        im_seg = Int8(vol_seg[:,i,:])
        diffuse_seg = anisotropic_diffusion(im_seg,niter=10,option=2).astype(np.float32)
        im_enhance = ContrastEnhance(diffuse_seg)
        otsu_th_opt = threshold_otsu(im_enhance)
        vol_opt[:,i,:] = np.uint8(im_enhance > otsu_th_opt)*255
        
        if verbose == True and i == idx:
            top = np.concatenate((im,im_seg),axis=1)
            bot = np.concatenate((vol_base_1[:,i,:],vol_base_2[:,i,:],vol_opt[:,i,:]),axis=1)
            
            plt.figure(figsize=(10,5))
            plt.axis('off')
            plt.title('slc:{}'.format(idx),fontsize=15)
            plt.imshow(top,cmap='gray')
#            plt.savefig("E:\\OCTA\\result\\vis.jpg")
            plt.show()
            
            plt.figure(figsize=(15,5))
            plt.axis('off')
            plt.imshow(bot,cmap='gray')
#            plt.savefig("E:\\OCTA\\result\\vis.jpg")
            plt.show()
            
    return vol_base_1, vol_base_2, vol_opt 

#%%
import warnings
warnings.filterwarnings("ignore")    

result_root = 'E:\\Fish\\test_result\\'
data_root = 'E:\\Fish\\test_data\\'

for i in range(1,7):
    vol_latent = util.nii_loader(result_root+'v30s{}_latent.nii'.format(i))
    vol_x = util.nii_loader(data_root+'v30s{}.nii.gz'.format(i))
    
    vol_base_1, vol_base_2, vol_opt = binarize(vol_x,vol_latent,True)
    
    util.nii_saver(vol_base_1,result_root,'v30s{}_bin_kmean.nii.gz'.format(i))
    util.nii_saver(vol_base_2,result_root,'v30s{}_bin_otsu.nii.gz'.format(i))
    util.nii_saver(vol_opt,result_root,'v30s{}_bin_LIFE.nii.gz'.format(i))
