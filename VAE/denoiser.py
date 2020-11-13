# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:37:05 2020

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

dataroot = 'E:\\OCTA\\data\\R=3\\train_data.pickle'
modelroot = 'E:\\Model\\'

batch_size = 2
n_epoch = 100
epoch_loss = []

dnoi_enc = (8,16,32,64,64)

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def criterion(y,y_syn):
    L1 = nn.L1Loss()
    L2 = nn.MSELoss()
    return L1(y_syn,y),0.01*L2(y_syn,y)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = arch.res_UNet(dnoi_enc).to(device)
#model.load_state_dict(torch.load(modelroot+'VAE_2.pt'))
#init_weights(model,'normal')

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
scheduler = StepLR(optimizer, step_size=3, gamma=0.2)

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
    for step,(tensor_x,tensor_y) in enumerate(train_loader):
        model.train()
        
        x = Variable(tensor_x).to(device)
        y = Variable(tensor_y).to(device)
        y_pred = model(x)
        
        l1,l2 = criterion(y,y_pred)
        loss = l1+l2
        sum_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 250 == 0:
            print('[%d/%d][%d/%d][L1:%.4f | L2:%.4f]'%(epoch,n_epoch,
                  step,len(train_loader),l1.item(),l2.item()))
        
        if step % 1000 == 0 and step != 0:
            im_dn = util.ImageRescale(y_pred[0,0,:,:].detach().cpu().numpy(),[0,255])
            im_x = util.ImageRescale(x[0,0,:,:].detach().cpu().numpy(),[0,255])
            im_y = util.ImageRescale(y[0,0,:,:].detach().cpu().numpy(),[0,255])
            
            plt.figure(figsize=(18,6))
            plt.axis('off')
            plt.title('Epoch: {}'.format(epoch),fontsize=15)
            plt.imshow(np.concatenate((im_x,im_dn,im_y),axis=1),cmap='gray')
            plt.show()
            
    epoch_loss.append(sum_loss)
    scheduler.step()
    
t2 = time.time()
print('time consume: {} min'.format((t2-t1)/60))

plt.figure(figsize=(10,8))
plt.title('Loss vs. Epoch',fontsize=15)
plt.plot(epoch_loss)
plt.show()

dnoi_model = 'denoiser.pt'
torch.save(model.state_dict(),modelroot+dnoi_model)