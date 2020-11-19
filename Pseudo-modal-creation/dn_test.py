# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:32:00 2020

@author: hudew
"""

import sys
sys.path.insert(0,'E:\\OCTA\\seg-VAE\\')
sys.path.insert(0,'E:\\tools\\')
import util
import VAE_arch as arch

import pickle,time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

dataroot = 'E:\\OCTA\\data\\pre_processed\\orig_proj(orig).pickle'
modelroot = 'E:\\Model\\'

dnoi_enc = (8,16,32,64,64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# orig -> proj{orig}
model_x = arch.res_UNet(dnoi_enc).to(device)
model_x.load_state_dict(torch.load(modelroot+'dn_orig_proj(orig).pt'))

# proj{orig} -> proj{var}
model_y = arch.res_UNet(dnoi_enc).to(device)
model_y.load_state_dict(torch.load(modelroot+'dn_proj(orig)_proj(var).pt'))

class dn_test_loader(Data.Dataset):
    
    def ToTensor(self, x, y):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        y_tensor = torch.tensor(y).type(torch.FloatTensor)
        y_tensor = torch.unsqueeze(y_tensor,dim=0)
        return x_tensor, y_tensor     
        
    def __init__(self,root):
        with open(root,'rb') as func:
            self.pair = pickle.load(func)
        
    def __len__(self):
        return len(self.pair)

    def __getitem__(self,idx):
        x, y = self.pair[idx]
#        y = util.ImageRescale(y,[0,1])
        x_tensor, y_tensor = self.ToTensor(x,y)
        return x_tensor, y_tensor

test_loader = Data.DataLoader(dataset=dn_test_loader(dataroot),
                               batch_size=1, shuffle=False)
   
#%%
vae_train_data = ()
for step,(tensor_x,tensor_y) in enumerate(test_loader):
    with torch.no_grad():
        x = Variable(tensor_x).to(device)
        y = Variable(tensor_y).to(device)
        
        dn_x = model_x(x)
        dn_y = model_y(y)
        
        im_x = util.ImageRescale(dn_x[0,0,:,:].detach().cpu().numpy(),[0,255])
        im_y = util.ImageRescale(dn_y[0,0,:,:].detach().cpu().numpy(),[0,255])
        
        vae_train_data = vae_train_data + ((im_x,im_y),)
        
        if step % 500 == 0 and step != 0:
            plt.figure(figsize=(12,6))
            plt.axis('off')
            plt.imshow(np.concatenate((im_x,im_y),axis=1),cmap='gray')
            plt.show()

with open('E:\\OCTA\\data\\VAE_train_data.pickle','wb') as func:
    pickle.dump(vae_train_data,func)
        
