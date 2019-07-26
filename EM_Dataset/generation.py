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

def convert (folder_im_png,folder_ma_png,) :
	print('converting datatset')
	num_batch=len(os.listdir(path_glob+folder_im_png))*2
	data_gen_args = dict()
	myGenerator = d.trainGenerator(path_glob,folder_im_png,folder_ma_png,data_gen_args,save_to_dir = path_glob+'image_tmp/')
	for i,batch in enumerate(myGenerator):
		if(i >= num_batch):
			break

if __name__ == '__main__':
	convert('data_png','label_png')
