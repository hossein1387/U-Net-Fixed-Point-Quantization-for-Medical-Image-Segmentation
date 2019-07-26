import os
import glob
import skimage.io as io
import skimage.transform as trans
import shutil
import numpy as np
from PIL import Image
#same as before 
path_glob='/Users/lurou_admin/Desktop/pancreas_test/'

def tri_im():
	print('sorting image and mask')
	t=os.listdir(path_glob+'image_tmp/')
	os.mkdir(path_glob+'img_512')
	os.mkdir(path_glob+'mask_512')

	for i in range  (len(t)):
	    if t[i][:4]=='imag':
	        shutil.move(path_glob+'image_tmp/'+t[i],path_glob+'img_512/'+t[i][6:])
	    elif t[i][:4]=='mask':
	        shutil.move(path_glob+'image_tmp/'+t[i],path_glob+'mask_512/'+t[i][5:])

def get_roi():
	print('getting region of interest')

	path_img=path_glob+'img_512/'
	path_ma=path_glob+'mask_512/'

	#creating folder that will contain the unique images (in case of bug in the previous program)

	os.mkdir(path_glob+'img_roi')
	os.mkdir(path_glob+'label_roi')

	path_roi=path_glob+'img_roi/'
	path_label_roi=path_glob+'label_roi/'


	t=os.listdir(path_img)
	u=os.listdir(path_ma)

	lis_unique=[]
	a=[]
	for i in range (len(t)):
		chiffre=t[i].split('_')
		if chiffre[0] in a :
			pass
		else:
			lis_unique.append(t[i])
			a.append(chiffre[0])

	if '.DS_Store' in lis_unique:
		lis.remove('.DS_Store')

	for x in lis_unique :
		shutil.move(path_img+x,path_roi+x)
		shutil.move(path_ma+x,path_label_roi+x)

	v=os.listdir(path_roi)
	if '.DS_Store' in v:
	 	v.remove('.DS_Store')
	print(v)

	for i in range (len(v)):
		ma=Image.open(path_label_roi+v[i])
		ct=Image.open(path_roi+v[i])
		arrma=np.array(ma)
		nz=np.nonzero(arrma)
		if nz!=0:
			index=int(np.round(len(nz[0])/2))
			w=nz[0][index]
			h=nz[1][index]
			print(t[i])
			if h+88>512 or w+56>512:
				print('danger problematic size'+v[i])
			elif h<88 or w<56:
				print('danger problematic size'+v[i])
			else:
				cropbox=[h-88,w-56,h+88,w+56]
				macrop=ma.crop(cropbox)
				ctcrop=ct.crop(cropbox)
				macrop.save(path_label_roi+v[i],'PNG')
				ctcrop.save(path_roi+v[i],'PNG')
	return('done')

def split val_train():
	w=os.listdir(path_glob+'img_roi/')
	w=w[int(np.round(len(w)*0.01)):]
	os.mkdir(path_glob+'val_img_roi')
	os.mkdir(path_glob+'val_label_roi')


	for x in w :
		shutil.move(path_glob+'img_roi'+x,path_glob+'val_img_roi'+x)
		shutil.move(path_glob+'label_roi'+x,path_glob+'val_label_roi'+x)

if __name__ == '__main__':
	tri_im()
	get_roi()


