from PIL import Image
import numpy as np
import scipy.misc
import imageio
import argparse
import cv2
import sys
import utility as util

red       = [255, 50 , 50 ]
green     = [0  , 255, 0  ]
blue      = [49 , 128, 255]
yellow    = [220, 240, 95 ]
orange    = [244, 161, 66 ]
white     = [255, 255, 255]
black     = [0  ,   0, 0  ]
burgundy  = [30 , 0  , 7  ]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_filename', help='input file name', required=True)
    parser.add_argument('-m', '--mask_filename', help='mask file name', required=True)
    parser.add_argument('-c', '--mask_color', help='overlay mask color', required=False, default="red")
    parser.add_argument('-v', '--mask_value', help='mask value', required=False, default="0")
    parser.add_argument('-o', '--output_image_name', help='output image name', required=False)
    parser.add_argument('-a', '--alpha', help='alpha', required=False, default=0.6, type=float)
    args = parser.parse_args()
    return vars(args)

def get_color(color_str):
    color_str = color_str.lower()
    if "red" in color_str:
        return red
    elif "yellow" in color_str:
        return yellow
    elif "white" in color_str:
        return white
    else:
        print("Error: Undefined color {0}".format(color_str))
        sys.exit()

def add_opacity(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 10 #creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def load_image( image_path ) :
    # import ipdb as pdb; pdb.set_trace()
    img = imageio.imread(image_path)
    data = np.asarray( img, dtype="int32" )
    return data

def color_mask(mask, color, mask_val):
    # import ipdb as pdb; pdb.set_trace()
    if np.shape(mask.shape)[0] == 2 : mask = to_rgb(mask)
    for irow, rows in enumerate(mask):
        for icol, cols in enumerate(rows):
           # import ipdb as pdb; pdb.set_trace()
            if (mask[irow][icol] != [mask_val,mask_val,mask_val]).any():
                # import ipdb as pdb; pdb.set_trace()
                mask[irow][icol] = [color[0], color[1], color[2]]
                # import ipdb as pdb; pdb.set_trace()
                # color_trans = make_rgb_transparent(color)
    return mask

def combine(img1, img2, img_color1, img_color2, union_color, orig_img, bkgnd_color):
    out_img = np.zeros(img1.shape)
    out_img = to_rgb(out_img)
    for irow, rows in enumerate(img1):
        for icol, cols in enumerate(rows):
            # import ipdb as pdb; pdb.set_trace()
            img1_cond = not (img1[irow][icol] == bkgnd_color).all()
            img2_cond = not (img2[irow][icol] == bkgnd_color).all()
            if img1_cond or img2_cond:
                if img1_cond and img2_cond:
                    out_img[irow][icol] = union_color
                elif img1_cond:
                    out_img[irow][icol] = img_color1
                else:
                    out_img[irow][icol] = img_color2
            else:
                out_img[irow][icol] = orig_img[irow][icol]
    return out_img

def lay_over(mask, orig_img, color, alpha, mask_val):
    # import ipdb as pdb;pdb.set_trace()
    if np.shape(mask.shape)[0] == 2 : mask = to_rgb(mask)
    if np.shape(orig_img.shape)[0] == 2 : orig_img = to_rgb(orig_img)
    mask = color_mask(mask=mask, color=color, mask_val=mask_val)
    cv2.addWeighted(mask, alpha, orig_img, alpha, 0, orig_img)
    return orig_img

def to_rgb(gray_img):
    # import ipdb as pdb; pdb.set_trace()
    img_size = gray_img.shape
    rgb_img  = np.zeros((img_size[0], img_size[1], 3))
    rgb_img[:,:,0]  = gray_img
    rgb_img[:,:,1]  = gray_img
    rgb_img[:,:,2]  = gray_img
    return rgb_img
if __name__ == '__main__':
    args = parse_args()
    input_filename = args['input_filename']
    mask_filename  = args['mask_filename']
    mask_color     = get_color(args['mask_color'])  
    mask_val       = int(args['mask_value'])
    image_name     = input_filename.split(".")[0]
    output_image   = args['output_image_name']
    alpha          = args['alpha']
    if output_image is None:
        output_image = image_name + "_over_layed" + ".png"
#    import ipdb as pdb; pdb.set_trace()
    img_in         = load_image(input_filename)
    img_mask       = load_image(mask_filename)

    util.save_image_to_path(lay_over(img_mask, img_in, mask_color, alpha, mask_val), output_image, "./")

