from __future__ import print_function
import sys
import cv2
import nibabel as nib
import data_prep as d
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

path_glob='/Users/lurou_admin/Desktop/pancreas_test/'

# retrieves relevant images and mask. Inputs , folder of label and folder of nifti files. Label files are named label00xx with xx number of the pancreas (as in the dataset)
#nifti files are named 00xx after cob=nversion to nifti
folder_image='data'
folder_mask='label'

def retrieve_ds(folder_im,folder_ma):
	print('retrieving datatset')
	path_ct=path_glob+folder_im

	path_label=path_glob+folder_ma

	path_out_label=path_label+'_png/'

	path_out_ct=path_ct+'_png/'

	os.mkdir(path_out_ct)
	os.mkdir(path_out_label)
	t=os.listdir(path_ct)
	if '.DS_Store' in t:
		t.remove('.DS_Store')
	for i in range (len(t)):
		ct=nib.load(path_ct+'/'+t[i])
		mask=nib.load(path_label+'/label'+t[i])
		img = np.array(ct.dataobj)
		l=np.array(mask.dataobj)
		for j in range (l.shape[2]):
			label=l[:,:,j]
			label=label.astype('int32')
			scan=img[:,:,j]
			scan=scan.astype('int32')
			if np.count_nonzero(label)==0:
				pass
			else:
				label_im= Image.fromarray(label)
				label_im.save(path_out_label+str(i)+'_'+str(j)+'.png')
				scan_im=Image.fromarray(scan)
				scan_im.save(path_out_ct+str(i)+'_'+str(j)+'.png')

	print('done')

	return(folder_im+'_png',folder_ma+'_png')


def convert (folder_im_png,folder_ma_png,) :
	print('converting datatset')
	num_batch=len(os.listdir(path_glob+folder_im_png))*2
	data_gen_args = dict()
	myGenerator = d.trainGenerator(path_glob,folder_im_png,folder_ma_png,data_gen_args,save_to_dir = path_glob+'image_tmp/')
	for i,batch in enumerate(myGenerator):
		if(i >= num_batch):
			break

if __name__ == '__main__':
	os.mkdir(path_glob+'image_tmp')
	a,b=retrieve_ds('data','label')
	z=len(os.listdir(path_glob+a)*2)
	sys.exit(a)
	#print(a,b)
	#convert(a,b)


