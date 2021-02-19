import sys
sys.path.insert(0,'E:\\tools\\')
import util
import numpy as np
import matplotlib.pyplot as plt
import os, imageio

root = 'E:\\Fish\\'

# iterate through different fish
for item in os.listdir(root):
    if not (np.int(item)>=27 and np.int(item)<=41):
        raise NameError('file not exist.')
    item_root = root+item+'\\'
    
    # iterate through sessions
    for session in os.listdir(item_root):
        session_root = item_root+session+'\\'
        
        # left & right eye
        for vol_name in os.listdir(session_root):
            if vol_name.endswith('.tif'):
                print('fish: {}, session: {}, eye: {}'.format(item,session[-1],vol_name[-5]))
                vol = imageio.volread(session_root+vol_name)
                util.nii_saver(vol,session_root,vol_name[:-4]+'.nii.gz')
