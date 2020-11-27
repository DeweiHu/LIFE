#!/usr/bin/env bash

Dir=/home/dewei/Desktop/octa/result
c3d=/home/dewei/src/c3d-1.0.0-Linux-x86_64/bin/c3d

vol=$Dir/vol_seg.nii.gz
vol_binary=$Dir/vol_binary.nii.gz
vol_opt=$Dir/vol_opt.nii.gz

$c3d vol -binarize -o vol_binary
$c3d vol_binary -comp -o vol_opt
