# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:00:51 2020

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
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

dataroot = 'E:\\OCTA\\data\\R=3\\train_data.pickle'
modelroot = 'E:\\Model\\'

batch_size = 1
n_epoch = 50
learning_rate = 0.001
epoch_loss = []

seg_enc = (4,16,32,64)
syn_enc = (4,16,32,64)
t = 3

def criterion(y,y_syn,mu,log_var):
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    KLD = -0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
    return L1(y_syn,y)+L2(y_syn,y)#+0.001*KLD
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = arch.VAE(seg_enc,syn_enc,t).to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

# x:[batch,n_channel,H,W], [0,255]
# y:[batch,n_channel,H,W], [0,1]
class HQ_human_train(Data.Dataset):
    
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
        x_tensor, y_tensor = self.ToTensor(x,y)
        return x_tensor, y_tensor

train_loader = Data.DataLoader(dataset=HQ_human_train(dataroot),
                               batch_size=batch_size, shuffle=True)

#%%
#for step,(x,y) in enumerate(train_loader):
#    pass
#print(x.size())
#print(y.size())

#%% 
print('training start...')
t1 = time.time()

for epoch in range(n_epoch):
    sum_loss = 0
    for step,(tensor_x,tensor_y) in enumerate(train_loader):
        model.train()
        
        x = Variable(tensor_x).to(device)
        y = Variable(tensor_y).to(device)
        y_seg,y_syn,mu,log_var = model(x)
        
        loss = criterion(y,y_syn,mu,log_var)
        sum_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print('[%d/%d][%d/%d][Loss: %.4f]'%(epoch,n_epoch,
                  step,len(train_loader),loss.item()))
        
        if step == len(train_loader)-1:
            seg = util.ImageRescale(y_seg[0,0,:,:].detach().cpu().numpy(),[0,255])
            syn = util.ImageRescale(y_syn[0,0,:,:].detach().cpu().numpy(),[0,255])
            im_x = util.ImageRescale(x[0,0,:,:].detach().cpu().numpy(),[0,255])
            im_y = util.ImageRescale(y[0,0,:,:].detach().cpu().numpy(),[0,255])
            
            top = np.concatenate((im_x,im_y),axis=1)
            bot = np.concatenate((seg,syn),axis=1)
            
            plt.figure(figsize=(12,12))
            plt.axis('off')
            plt.title('Epoch: {}'.format(epoch),fontsize=15)
            plt.imshow(np.concatenate((top,bot),axis=0),cmap='gray')
            plt.show()
            
    epoch_loss.append(sum_loss)

t2 = time.time()
print('time consume: {} min'.format((t2-t1)/60))

plt.figure(figsize=(10,8))
plt.title('Loss vs. Epoch',fontsize=15)
plt.plot(epoch_loss)
plt.show()



