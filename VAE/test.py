'''
import a volume with dimension [H,slc,W]
output 1: unthreshold segmentation (vol_seg)
output 2: intermediate synthesized volume (vol_syn) 
'''

import sys
sys.path.insert(0,'E:\\OCTA\\seg-VAE\\')
sys.path.insert(0,'E:\\tools\\')
import util
import VAE_arch as arch
import pickle,cv2,random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

global msk, loc_map
msk = (320,320)

'''
randomly sample [num] of masks to pass the model.
The result for the whole image is generated from all the masks
'''
def RandomCropper(im,num):
    global msk
    h,w = im.shape
    lib = ([0,0],[0,w-msk[1]],[h-msk[0],0],[h-msk[0],w-msk[1]])
    
    # random points
    for i in range(num):
        pseed = [random.randint(0,h-msk[0]),random.randint(0,w-msk[1])]
        lib = lib + (pseed,)
    
    if not len(lib) == num+4:
        raise ValueError('Length not matched.')
        
    # form a stack for model testing
    stack = np.zeros([len(lib),msk[0],msk[1]],dtype=np.float32)
    for i in range(len(lib)):
        x,y = lib[i] 
        stack[i,:,:] = util.ImageRescale(im[x:x+msk[0],y:y+msk[1]],[0,255])
    
    return stack, lib

'''
 h,w : the dimension of the original image
stack: the pre-segmented pieces
 loc : the location of each piece
'''
def merge(h,w,stack,loc):
    global msk
    opt = np.zeros([h,w],dtype=np.float32)
    counter = np.zeros([h,w],dtype=np.float32)

    for i in range(len(loc)):
        x,y = loc[i]
        opt[x:x+msk[0],y:y+msk[1]] += util.ImageRescale(stack[i,:,:],[0,255])
        counter[x:x+msk[0],y:y+msk[1]] += np.ones([msk[0],msk[1]],dtype=np.float32)
    opt = opt/counter
    return opt


class vae_test_loader(Data.Dataset):
    def ToTensor(self, x):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        return x_tensor
    
    def __init__(self,vol,num):
        # break the single slice into 4 pieces that of msk size
        global msk, loc_map
        loc_map = []
        self.vol = []
        self.h,self.slc,self.w = vol.shape
        for i in range(self.slc):
            stack, lib = RandomCropper(vol[:,i,:],num)
            loc_map.append(lib)
            for j in range(num+4):    
                self.vol.append(stack[j,:,:])
                
    def __len__(self):
        return len(self.vol)

    def __getitem__(self,idx):
        x = self.vol[idx]       # [num+4,1,h,w] 
        x_tensor = self.ToTensor(x)
        return x_tensor


#%%
def SegVAE(vol,num,model_dn,model_refine,model_vae):
    global msk, loc_map
    h,slc,w = vol.shape
    # set batchsize to num+4
    test_loader = Data.DataLoader(dataset=vae_test_loader(vol,num),
                                   batch_size=num+4, shuffle=False)
    
    # define the output volumes
    vol_dn = np.zeros(vol.shape,dtype=np.float32)
    vol_seg = np.zeros(vol.shape,dtype=np.float32)
    vol_syn = np.zeros(vol.shape,dtype=np.float32)

    for step,tensor_x in enumerate(test_loader):
        loc = loc_map[step]
        # stack size: [4,0,320,320]
        x = Variable(tensor_x).to(device)
        dn_x = model_dn(x)
        _,stack_syn = model_refine(dn_x)
        stack_seg,_ = model_vae(stack_syn)
        
        stack_dn = dn_x[:,0,:,:].detach().cpu().numpy()
        stack_seg = -stack_seg[:,0,:,:].detach().cpu().numpy()
        stack_syn = stack_syn[:,0,:,:].detach().cpu().numpy()
        
        # combine num+4 pieces
        vol_dn[:,step,:] = merge(h,w,stack_dn,loc)
        vol_seg[:,step,:] = merge(h,w,stack_seg,loc)
        vol_syn[:,step,:] = merge(h,w,stack_syn,loc)

    return vol_dn, vol_seg, vol_syn

if __name__ == "__main__":

    dataroot = 'E:\\OCTA\\data\\R=3\\'
    modelroot = 'E:\\Model\\'
    saveroot = 'E:\\OCTA\\paper_img\\'
#
    dnoi_enc = (8,16,32,64,64)
    seg_enc = (8,16,32,64,64)
    syn_enc = (8,16,32)
    t = 3
    num = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # denoise: orig -> LF{orig} = dn{orig}
    model_dn = arch.res_UNet(dnoi_enc).to(device)
    model_dn.load_state_dict(torch.load(modelroot+'dn_orig_proj(orig).pt'))
    
    # contrast enhance: dn{orig} -> CE-LF{orig} = ce{orig}
    model_refine = arch.VAE(seg_enc,syn_enc,t).to(device)
    model_refine.load_state_dict(torch.load(modelroot+'vae_refine.pt'))
    
    # segmentation: seg{orig} -> LF{orig}
    model_vae = arch.VAE(seg_enc,syn_enc,t).to(device)
    model_vae.load_state_dict(torch.load(modelroot+'vae.pt'))

    #%% input volume
    vol = util.nii_loader(dataroot+'fovea5.nii.gz')
    vol_AR = util.nii_loader(dataroot+'AR_fovea5.nii.gz')
    vol = vol[30:433,30:40,:]
    vol_AR = vol_AR[30:433,30:40,:]
    
    vol_dn,vol_seg,vol_syn = SegVAE(vol_AR,num,model_dn,model_refine,model_vae)
    util.nii_saver(vol_AR,saveroot,'orig5.nii.gz')
    util.nii_saver(vol_dn,saveroot,'vol_dn5.nii.gz')
    util.nii_saver(vol_seg,saveroot,'vol_seg5.nii.gz')
    util.nii_saver(vol_syn,saveroot,'vol_syn5.nii.gz')
