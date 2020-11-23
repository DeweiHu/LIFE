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
from torch.nn import init

dataroot = 'E:\\OCTA\\data\\VAE_train_data.pickle'
modelroot = 'E:\\Model\\'

batch_size = 2
n_epoch = 150
epoch_loss = []

seg_enc = (8,16,32,64,64)
syn_enc = (8,16,32)
t = 3
    
def criterion(y,y_syn):
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    return L1(y_syn,y),0.02*L2(y_syn,y)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = arch.VAE(seg_enc,syn_enc,t).to(device)

#%%
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

Seg_opt = torch.optim.Adam(model.Seg_Net.parameters(),lr=2e-3)
Seg_sch = StepLR(Seg_opt, step_size=3, gamma=0.5)

Syn_opt = torch.optim.Adam(model.Syn_Net.parameters(),lr=1e-4)
Syn_sch = StepLR(Seg_opt, step_size=3, gamma=0.5)

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
#        y = util.ImageRescale(y,[0,1])
        x_tensor, y_tensor = self.ToTensor(x,y)
        return x_tensor, y_tensor

train_loader = Data.DataLoader(dataset=HQ_human_train(dataroot),
                               batch_size=batch_size, shuffle=True)

#%% 
print('training start...')
t1 = time.time()

for epoch in range(n_epoch):
    sum_loss = 0
    print('SegNet lr:{}, SynNet lr:{}'.format(Seg_sch.get_lr()[0],Syn_sch.get_lr()[0]))
    for step,(tensor_x,tensor_y) in enumerate(train_loader):
        model.train()
        
        x = Variable(tensor_x).to(device)
        y = Variable(tensor_y).to(device)
        y_seg,y_syn = model(x)
        
        l1,l2 = criterion(y,y_syn)
        loss = l1+l2
        sum_loss += loss
        
        if epoch <= 3:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            Seg_opt.zero_grad()
            Syn_opt.zero_grad()
            
            loss.backward()
            
            Seg_opt.step()
            Syn_opt.step()

        if step % 250 == 0:
            print('[%d/%d][%d/%d][L1:%.4f | L2:%.4f]'%(epoch,n_epoch,
                  step,len(train_loader),l1.item(),l2.item()))
        
        if step % 1000 == 0 and step != 0:
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
    
    if epoch > 3:
        Seg_sch.step()
        Syn_sch.step()
    
t2 = time.time()
print('time consume: {} min'.format((t2-t1)/60))

plt.figure(figsize=(10,8))
plt.title('Loss vs. Epoch',fontsize=15)
plt.plot(epoch_loss)
plt.show()

VAE_model = 'vae_dn.pt'
torch.save(model.state_dict(),modelroot+VAE_model)


    
