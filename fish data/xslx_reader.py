# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:28:01 2021

@author: hudew
"""

import pandas as pd
import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import matplotlib.pyplot as plt
import os, pickle

def GetRange(left):
    if not left == '':
        semic_idx = [pos for pos, char in enumerate(left) if char == ':']
        if len(semic_idx) == 1:
            lb = np.int(left[:semic_idx[0]])
            hb = np.int(left[semic_idx[0]+1:])
            rng_left = [lb,hb]
            
        elif len(semic_idx) == 2:
            comma_idx = left.find(',')
            
            lb1 = np.int(left[:semic_idx[0]])
            hb1 = np.int(left[semic_idx[0]+1:comma_idx])
            
            lb2 = np.int(left[comma_idx+1:semic_idx[1]])
            hb2 = np.int(left[semic_idx[1]+1:])
            
            rng_left = [lb1,hb1,lb2,hb2]
        else:
            raise NotImplementedError
    else:
        rng_left = []
    return rng_left

def string2idx(lib):
    # index of start & end of parenthesis
    paren_start = [pos for pos, char in enumerate(lib) if char == '[']
    paren_end = [pos for pos, char in enumerate(lib) if char == ']']
    
    # index of chosen range for left & right volume
    left = lib[paren_start[0]+1:paren_end[0]]
    right = lib[paren_start[1]+1:paren_end[1]]
    
    # find semicolumn
    rng_left = GetRange(left)
    rng_right = GetRange(right)
    
    return rng_left, rng_right
        
def GetAtlas(vol,lb,hb,r,opt):
    for i in range(lb+r,hb-r):
        x = vol[:,i,200:700]
        y = np.mean(vol[:,i-r:i+r+1,200:700],axis=1)
        opt = opt + ((x,y),)
    return opt

#%% main
root = 'E:\\OCTA\\fish data\\'
dataroot = 'E:\\Fish\\'
file = 'fish.csv'

df = pd.read_csv(root+file)
row, col = df.shape
opt = ()
r = 3

for i in range(row):
    # index 0 in col indicate ID of the fish
    fish_root = dataroot+np.str(df.loc[i][0])+'\\'
    print('fish: {}, samples: {}'.format(df.loc[i][0],len(opt)))
    # get session number
    for session in os.listdir(fish_root):
        session_root = fish_root+session+'\\'
        j = np.int(session[-1])
        
        for item in os.listdir(session_root):
            if item.endswith('L.nii.gz'):
                vol_L = util.nii_loader(session_root+item)
            elif item.endswith('R.nii.gz'):
                vol_R = util.nii_loader(session_root+item)
                
        if type(df.loc[i][j]) == str:
            lib = df.loc[i][j]
            rng_left, rng_right = string2idx(lib)
            
            if len(rng_left) == 2:
                opt = GetAtlas(vol_L,rng_left[0],rng_left[1],r,opt)
            elif len(rng_left) == 4:
                opt = GetAtlas(vol_L,rng_left[0],rng_left[1],r,opt)
                opt = GetAtlas(vol_L,rng_left[2],rng_left[3],r,opt)
            
            if len(rng_right) == 2:
                opt = GetAtlas(vol_R,rng_right[0],rng_right[1],r,opt)
            elif len(rng_right) == 4:
                opt = GetAtlas(vol_R,rng_right[0],rng_right[1],r,opt)
                opt = GetAtlas(vol_R,rng_right[2],rng_right[3],r,opt)
                
#%%
with open(root+'fish_data.pickle','wb') as func:
    pickle.dump(opt,func)

        
        
        
        
            