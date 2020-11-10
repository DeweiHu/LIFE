# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:46:42 2020

@author: hudew
"""

import torch
import torch.nn as nn

#%% Seg-Net: R2U-Net || Syn-Net: Res-UNet
class Recurrent_block(nn.Module):
    def __init__(self, nch, t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.nch = nch
        self.conv = nn.Sequential(
                nn.Conv2d(nch,nch,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(nch),
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class Residual_block(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Residual_block,self).__init__()
        self.align = nn.Conv2d(in_channels = nch_in,
                               out_channels = nch_out,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0)
        self.dualconv = nn.Sequential(
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out),
                nn.ELU(),
                nn.Conv2d(in_channels = nch_out,
                          out_channels = nch_out,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1),
                nn.BatchNorm2d(num_features = nch_out)
                )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.align(x)
        x1 = self.dualconv(x)
        opt = self.relu(torch.add(x,x1))
        return opt

# recurrent-residual block
class RR_block(nn.Module):
    def __init__(self, nch_in, nch_out, t=2):
        super(RR_block,self).__init__()
        self.dualrec = nn.Sequential(
                Recurrent_block(nch_out,t=t),
                Recurrent_block(nch_out,t=t)
                )
        self.align = nn.Conv2d(nch_in,nch_out,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.align(x)
        x1 = self.dualrec(x)
        opt = self.relu(torch.add(x,x1))
        return opt

class R2U_Net(nn.Module):
    def __init__(self, nch_enc, t=2):
        super(R2U_Net,self).__init__()
        
        # (assume input_channel=1)
        self.nch_in = 1
        self.nch_enc = nch_enc
        self.nch_aug = (self.nch_in,)+self.nch_enc
        
        # module list
        encoder = []
        trans_down = []
        decoder = []
        trans_up= []
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            encoder.append(RR_block(self.nch_aug[i],self.nch_aug[i+1]))
            trans_down.append(self.trans_down(self.nch_enc[i],self.nch_enc[i]))
            # decoder & upsample
            trans_up.append(self.trans_up(self.nch_enc[-1-i],self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                decoder.append(RR_block(self.nch_aug[-1-i]*2,1))
            else:
                decoder.append(RR_block(self.nch_aug[-1-i]*2,self.nch_aug[-2-i]))
        
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.trans_down = nn.ModuleList(trans_down)
        self.trans_up = nn.ModuleList(trans_up)
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.trans_down[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.trans_up[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred
            
    def trans_down(self, nch_in, nch_out):
        return nn.Sequential(
                nn.Conv2d(in_channels=nch_in, 
                          out_channels=nch_out, 
                          kernel_size=4,
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(nch_out),
                nn.ELU()
                )
                
    def trans_up(self,nch_in,nch_out):
        return nn.Sequential(
                nn.ConvTranspose2d(in_channels=nch_in, 
                                   out_channels=nch_out,
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1),
                nn.BatchNorm2d(nch_out),
                nn.ELU()
                )

class res_UNet(nn.Module):
    def __init__(self, nch_enc):
        super(res_UNet,self).__init__()
        
        # (assume input_channel=1)
        self.nch_in = 1
        self.nch_enc = nch_enc
        self.nch_aug = (self.nch_in,)+self.nch_enc
        
        # module list
        encoder = []
        trans_down = []
        decoder = []
        trans_up= []
        
        for i in range(len(self.nch_enc)):
            # encoder & downsample
            encoder.append(Residual_block(self.nch_aug[i],self.nch_aug[i+1]))
            trans_down.append(self.trans_down(self.nch_enc[i],self.nch_enc[i]))
            # decoder & upsample
            trans_up.append(self.trans_up(self.nch_enc[-1-i],self.nch_enc[-1-i]))
            if i == len(self.nch_enc)-1:
                decoder.append(Residual_block(self.nch_aug[-1-i]*2,1))
            else:
                decoder.append(Residual_block(self.nch_aug[-1-i]*2,self.nch_aug[-2-i]))
        
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.trans_down = nn.ModuleList(trans_down)
        self.trans_up = nn.ModuleList(trans_up)
    
    def forward(self, x):
        cats = []
        # encoder
        for i in range(len(self.nch_enc)):
            layer_opt = self.encoder[i](x)
            x = self.trans_down[i](layer_opt)
            cats.append(layer_opt)
        
        # bottom layer
        layer_opt = x
        
        # decoder
        for i in range(len(self.nch_enc)):
            x = self.trans_up[i](layer_opt)
            x = torch.cat([x,cats[-1-i]],dim=1)
            layer_opt = self.decoder[i](x)

        y_pred = layer_opt
        return y_pred
            
    def trans_down(self, nch_in, nch_out):
        return nn.Sequential(
                nn.Conv2d(in_channels=nch_in, 
                          out_channels=nch_out, 
                          kernel_size=4,
                          stride=2, 
                          padding=1),
                nn.BatchNorm2d(nch_out),
                nn.ELU()
                )
                
    def trans_up(self,nch_in,nch_out):
        return nn.Sequential(
                nn.ConvTranspose2d(in_channels=nch_in, 
                                   out_channels=nch_out,
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1),
                nn.BatchNorm2d(nch_out),
                nn.ELU()
                )
                
#%% VAE
class VAE(nn.Module):
    def __init__(self, seg_enc, syn_enc, t=2):
        super(VAE,self).__init__()
        self.seg_enc = seg_enc
        self.t = t
        self.syn_enc = syn_enc
        self.bifurcator = nn.Conv2d(in_channels = 1,
                                    out_channels = 2,
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0)
        # Encoder: Seg_Net, Decoder: Syn_Net
        self.Seg_Net = R2U_Net(self.seg_enc,self.t)
        self.Syn_Net = res_UNet(self.syn_enc)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
    
    def forward(self, x):
        # Encoder
        Seg_opt = self.Seg_Net(x)           # [batch,channel=1,H,W]
        latent = self.bifurcator(Seg_opt)   # [batch,channel=2,H,W]
        mu = latent[:,0,:,:]
        log_var = latent[:,1,:,:]
        
        # Reparameterize
        z = self.reparameterize(mu,log_var)
        
        # Decoder
        Syn_opt = self.Syn_Net(z)
        
        return Seg_opt, Syn_opt
        
    