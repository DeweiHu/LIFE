#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 19:58:50 2020

@author: hud4
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift

shifted, error, diffphase = register_translation(fix,mov,100)
reg = shift(mov,shift=(shifted[0], shifted[1]), mode='constant')





