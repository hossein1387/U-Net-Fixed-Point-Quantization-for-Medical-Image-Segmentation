import os
import glob
import skimage.io as io
import skimage.transform as trans
import shutil
import numpy as np
from PIL import Image

path_glob='/Users/lurou_admin/Desktop/EM_dataset/'

def tri_im():
	print('sorting image and mask')
	t=os.listdir(path_glob+'aug/')
	os.mkdir(path_glob+'img_256')
	os.mkdir(path_glob+'mask_256')

	for i in range  (len(t)):
		if t[i][:4]=='imag':
			shutil.move(path_glob+'aug/'+t[i],path_glob+'img_256/'+t[i][6:])
		elif t[i][:4]=='mask':
			shutil.move(path_glob+'aug/'+t[i],path_glob+'mask_256/'+t[i][5:])

def crop():
	pth=path_glob+'img_256/'
	v=os.listdir(pth)
	os.mkdir(path_glob+'im_ready')
	os.mkdir(path_glob+'ma_ready')
	out_im=path_glob+'im_ready/'
	out_ma=path_glob+'ma_ready'

		if '.DS_Store' in v:
		 	v.remove('.DS_Store')
		#print(v)
		cropbox=[28,28,228,228]

		for i in range (len(v)):
			ma=Image.open(path_glob+'label_256/'+v[i])
			ct=Image.open(pth+v[i])		
			macrop=ma.crop(cropbox)
			ctcrop=ct.crop(cropbox)
			macrop.save(out_im+v[i],'PNG')
			ctcrop.save(ou_ma+v[i],'PNG')

def split val_train():
	w=os.listdir(path_glob+'im_ready/')
	w=w[int(np.round(len(w)*0.01)):]
	os.mkdir(path_glob+'val_img')
	os.mkdir(path_glob+'val_label')


	for x in w :
		shutil.move(path_glob+'im_ready/'+x,path_glob+'val_img/'+x)
		shutil.move(path_glob+'ma_ready/'+x,path_glob+'val_label/'+x)

if __name__ == '__main__':
	tri_im()
	crop()
	split_val_train()


