import util
import cv2, subprocess
import numpy as np 

def cca(vol_seg,dir,size_th=30):
	util.nii_saver(vol_seg,dir,'vol_seg.nii.gz')
	subprocess.call("/home/dewei/tool/cca.sh")
	vol_binary = util.nii_loader(dir+'vol_binary.nii.gz')
	vol_cca = util.nii_loader(dir+'cca.nii.gz')

	num_component = int(vol_cca.max())
	hist = np.histgram(vol_cca,num_component)
	size_vec = hist[0]

	cca_th = np.sum(np.uint8(size_vec>size_th))-1
	h,d,w = vol_cca.shape
	vol_opt = np.zeros(vol_cca.shape,dtype=np.int)

	for i in range(h):
		for j in range(d):
			for k in range(w):
				if (vol_cca[i,j,k]<=cca_th and vol_cca[i,j,k]>0):
					vol_opt[i,j,k] = 1

	return vol_opt


if __name__=="__main__":
	dataroot = '/home/dewei/Desktop/octa/result/'
	# vol_seg is the output of Otsu's method
	vol_seg = util.nii_loader(dataroot+'vol_seg.nii.gz')
	vol_opt = cca(vol_seg,dataroot,30)
	