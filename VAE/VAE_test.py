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
import pickle,cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

global msk
msk = (320,320)

class vae_test_loader(Data.Dataset):
    def ToTensor(self, x):
        x_tensor = torch.tensor(x).type(torch.FloatTensor)
        x_tensor = torch.unsqueeze(x_tensor,dim=0)
        return x_tensor
    def __init__(self,vol):
        # break the single slice into 4 pieces that of msk size
        global msk
        self.vol = []
        self.h,self.slc,self.w = vol.shape
        for i in range(self.slc):
            self.vol.append(vol[:msk[0],:msk[1]],vol[self.h-msk[0]:,:msk[1]],
                vol[:msk[0],self.w-msk[1]:],vol[self.h-msk[0]:,self.w-msk[1]:])
    
    def __len__(self):
        return self.slc

    def __getitem__(self,idx):
        x = self.vol[idx]
        x_tensor = self.ToTensor(x)
        return x_tensor

def otsu(im):
    im_uint = cv2.normalize(src=im,dst=0,alpha=0,beta=255,
                            norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    _,im_binary = cv2.threshold(im_uint,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return im_binary

def SegVAE(vol,model_dn,model_refine,model_vae):
    global msk
    h,slc,w = vol.shape
    # set batchsize to 4 so that 4 pieces are in the same stack
    test_loader = Data.DataLoader(dataset=vae_test_loader(vol),
                                   batch_size=4, shuffle=False)
    
    # define the output volumes
    vol_seg = np.zeros(vol.shape,dtype=np.float32)
    vol_syn = np.zeros(vol.shape,dtype=np.float32)

    for step,(tensor_x) in enumerate(test_loader):
        # stack size: [4,0,320,320]
        x = Variable(tensor_x).to(device)
        dn_x = model_dn(x)
        _,stack_syn = model_refine(dn_x)
        stack_seg,_ = model_vae(im_syn)

        stack_seg = stack_seg.detach().cpu().numpy()
        stack_syn = stack_syn.detach().cpu().numpy()

        # combine 4 pieces
        for i in range(slc):
            vol_seg[:msk[0],i,:msk[1]] = util.ImageRescale(stack_seg[0,0,:,:],[0,255])
            vol_seg[h-msk[0]:,i,:msk[1]] = util.ImageRescale(stack_seg[1,0,:,:],[0,255])
            vol_seg[:msk[0],i,w-msk[1]:] = util.ImageRescale(stack_seg[2,0,:,:],[0,255])
            vol_seg[h-msk[0]:,i,w-msk[0]:] = util.ImageRescale(stack_seg[3,0,:,:],[0,255])

            vol_syn[:msk[0],i,:msk[1]] = util.ImageRescale(stack_syn[0,0,:,:],[0,255])
            vol_syn[h-msk[0]:,i,:msk[1]] = util.ImageRescale(stack_syn[1,0,:,:],[0,255])
            vol_syn[:msk[0],i,w-msk[1]:] = util.ImageRescale(stack_syn[2,0,:,:],[0,255])
            vol_syn[h-msk[0]:,i,w-msk[0]:] = util.ImageRescale(stack_syn[3,0,:,:],[0,255])

    return vol_seg, vol_syn

if __name__ == "__main__":

    dataroot = 'E:\\OCTA\\data\\pre_processed\\'
    modelroot = 'E:\\Model\\'
#
    dnoi_enc = (8,16,32,64,64)
    seg_enc = (8,16,32,64,64)
    syn_enc = (8,16,32)
    t = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # orig -> proj{orig}
    model_dn = arch.res_UNet(dnoi_enc).to(device)
    model_dn.load_state_dict(torch.load(modelroot+'dn_orig_proj(orig).pt'))
    # proj{orig} -> proj{var}
    model_refine = arch.VAE(seg_enc,syn_enc,t).to(device)
    model_refine.load_state_dict(torch.load(modelroot+'vae_refine.pt'))
    # proj{orig} -> proj{var}
    model_vae = arch.VAE(seg_enc,syn_enc,t).to(device)
    model_vae.load_state_dict(torch.load(modelroot+'vae.pt'))

    #%% input volume
    vol = util.nii_loader(dataroot+'orig_fovea5.nii.gz')
    vol = vol[,15:35,:]

    vol_seg,vol_syn = SegVAE(vol,model_dn,model_refine,model_vae)

