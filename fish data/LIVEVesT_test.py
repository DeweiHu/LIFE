# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 23:37:34 2021

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\OCTA\\seg-VAE\\')
sys.path.insert(0,'E:\\tools\\')
import util
import VAE_arch as arch

import pickle, cv2
import numpy as np
import matplotlib.pyplot as plt

import torch, os
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from PIL import Image, ImageEnhance

dataroot = 'E:\\Fish\\test_data\\'
modelroot = 'E:\\Model\\'

seg_enc = (8,16,32,64,64)
syn_enc = (8,16,32)
t = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = arch.VAE(seg_enc,syn_enc,t).to(device)
model1.load_state_dict(torch.load(modelroot+'f_try.pt'))

model2 = arch.VAE(seg_enc,syn_enc,t).to(device)
model2.load_state_dict(torch.load(modelroot+'Get_Latent.pt'))

#%% data loader
class test(Data.Dataset):
    
    def ToTensor(self, x):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        return x_tensor  
    
    def __init__(self,root):
        self.data = []
        self.vol = util.nii_loader(root)
        _, self.slc, _ = self.vol.shape
        for i in range(self.slc):
            self.data.append(self.vol[:,i,:])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        x = self.data[idx]
        x_tensor = self.ToTensor(x)
        return x_tensor

#%% main

for file in os.listdir(dataroot):
    # load volume as dataloader
    if file.endswith('.nii.gz'):
        test_loader = Data.DataLoader(dataset=test(dataroot+file),batch_size=1,
                                      shuffle=False)
        print('volume {} loaded.'.format(file))
    
    vol_latent = np.zeros([480,len(test_loader),480],dtype=np.float32)
    vol_syn = np.zeros([480,len(test_loader),480],dtype=np.float32)
    
    # test
    for step, tensor_x in enumerate(test_loader):
        x = Variable(tensor_x).to(device)
        # take the synthesized image as denoised x
        _,dn_x = model1(x)
        latent,syn = model2(dn_x)
        
        im_x = x[0,0,:,:].detach().cpu().numpy()
        im_x = util.ImageRescale(im_x,[0,255])
        
        # latent image
        im_latent = latent[0,0,:,:].detach().cpu().numpy()
        im_latent = util.ImageRescale(-im_latent,[0,255])
        
        # synthetic image
        im_syn = syn[0,0,:,:].detach().cpu().numpy()
        im_syn = util.ImageRescale(im_syn,[0,255])
        
        vol_latent[:,step,:] = im_latent
        vol_syn[:,step,:] = im_syn
        
        if step % 5 == 0:
            plt.figure(figsize=(15,5))
            plt.imshow(np.concatenate((im_x,im_latent,im_syn),axis=1),cmap='gray')
            plt.axis('off')
            plt.show()

        util.nii_saver(vol_latent,'E:\\Fish\\test_result\\',file[:-7]+'_latent.nii')
        util.nii_saver(vol_syn,'E:\\Fish\\test_result\\',file[:-7]+'_syn.nii')
        







