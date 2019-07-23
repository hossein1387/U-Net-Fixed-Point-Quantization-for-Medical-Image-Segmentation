import sys
sys.path.insert(0, '../UNET_NIH_DATASET/UNET_QUANT_NIH_BASE/')
from datasets import *
from models import *
from transforms import *
from losses import *
from metrics import *
from filters import *
import numpy as np
import utility
import torch
import models as m
from torchsummary import summary
import argparse
import config as configuration
from torch.utils.data import DataLoader
from PIL import Image
import layover as lay
from random import randint
import natsort 
import glob

# rand_indx = randint(0, 3)
global is_cuda 

def parse_args():
    ex = "python get_nih_result.py -b ../UNET_NIH_DATASET/ -c UNET_NIH_DATASET_FULLPRECISION -s 0"
    parser = argparse.ArgumentParser(ex)
    parser.add_argument('-b', '--base_nih_dir', help='path to nih experiment folder', required=False, default="/users/hemmat/MyRepos/UNET_Quantized/UNET_NIH_DATASET/")
    parser.add_argument('-c','--configs', nargs='+', help='configurations to be used', required=True)
    parser.add_argument('-s','--seed', help='manual seed', required=False, default=0, type=int)
    parser.add_argument('-e','--export_onnx', help='export to onnx', action='store_true')
    args = parser.parse_args()
    return vars(args)

def create_mask(img, mask_val, back_color):
    new_img = np.zeros(img.shape) 
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if np.abs(pixel - back_color) > 10:
                new_img[i][j] = mask_val
            else:
                new_img[i][j] = back_color
    new_img = 255 - new_img
    return new_img

def get_input_data(data_set_path="/export/tmp/hemmat/datasets/nih/pancreas/sliced/data/"):
    global is_cuda
    # import ipdb as pdb; pdb.set_trace()
    target_val_path   = natsort.natsorted(glob.glob(data_set_path + 'val_mask/*.png'))
    image_val_path    = natsort.natsorted(glob.glob(data_set_path + 'val_img/*.png'))
    gmdataset_val = EMdataset(image_paths=image_val_path, target_paths=target_val_path)
    val_loader = DataLoader(gmdataset_val, batch_size=4,
                            shuffle=True,
                            num_workers=1)
    for i, batch in enumerate(val_loader):
        input_samples, gt_samples, idx = batch[0], batch[1], batch[2]
        var_input = input_samples.cuda() if is_cuda else input_samples
        var_gt    = gt_samples.cuda() if is_cuda else gt_samples
        var_gt    = var_gt.float() if is_cuda else var_gt
        break
    return var_input, var_gt

########################################################################################
# depricated
# def get_images(config):
#     base_path= config['log_output_dir']+config['experiment_name'] +"/figs/"
#     in_img   = imageio.imread(base_path+"EM_UNET_ORIG_0200.png")
#     gt_img   = imageio.imread(base_path+"EM_UNET_GT_0200.png")
#     pred_img = imageio.imread(base_path+"EM_UNET_PRED_0200.png")
#     return in_img, gt_img, pred_img
########################################################################################

def get_model(config):
    global is_cuda
    if not is_cuda:
        device = 'cpu'
        unet_model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
        unet_model = unet_model.to(device)
        unet_model.load_state_dict(torch.load(config['trained_model'], map_location=lambda storage, loc: storage))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet_model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
        unet_model = unet_model.to(device)
        unet_model.load_state_dict(torch.load(config['trained_model']))
    return unet_model

def get_result(var_input, var_gt, model, config, rand_indx):
    # import ipdb as pdb; pdb.set_trace()
    preds = model(var_input)
    var_input = var_input.detach().cpu().numpy()[rand_indx][0]
    var_gt    = var_gt.detach().cpu().numpy()[rand_indx][0]
    preds     = preds.detach().cpu().numpy()[rand_indx][0]
    save_image(var_input, var_gt, preds, "./", config, rand_indx) 
    return

def save_image(in_img, gt_img, pred_img, path, config, rand_indx):
    # import ipdb as pdb; pdb.set_trace()
    img_name      = "NIH_UNET_"      + config['experiment_name'] + "_" +str(rand_indx) + ".png"
    orig_img_name = "NIH_UNET_ORIG_" + config['experiment_name'] + "_" +str(rand_indx) + ".png"
    gt_img_name   = "NIH_UNET_GT_"   + config['experiment_name'] + "_" +str(rand_indx) + ".png"
    pred_img_name = "NIH_UNET_PRED_" + config['experiment_name'] + "_" +str(rand_indx) + ".png"
    gt_img   *= 255
    pred_img *= 255
    in_img   *= 255
    gt_img    = create_mask(gt_img, 0, 255)
    pred_img  = create_mask(pred_img, 0, 255)
    # utility.save_image_to_path(in_img, orig_img_name, path)
    # utility.save_image_to_path(gt_img, gt_img_name, path)
    # utility.save_image_to_path(pred_img, pred_img_name, path)
    # gt_over_layed   = lay.lay_over(gt_img,   in_img,        lay.blue, alpha=0.5, mask_val=255)
    # pred_over_layed = lay.lay_over(pred_img, gt_over_layed, lay.red,    alpha=0.5, mask_val=255)
    # utility.save_image_to_path(pred_over_layed, img_name, path)
    pred_over_layed = lay.combine(gt_img, pred_img, lay.red, lay.blue, lay.orange, in_img, lay.white)
    utility.save_image_to_path(pred_over_layed, img_name, path)


def parse_configs(config_files, base_em_dir):
    global is_cuda
    is_cuda = True
    configs = []
    for config_file in config_files:
        config_file = base_em_dir+config_file+"/config.yaml"
        config = configuration.Configuration(model_type, config_file)
        config = config.config_dict
        is_cuda &= True if (str(config['gpu_core_num']).lower() != "none" and torch.cuda.is_available()) else False
        model  = get_model(config)
        config_obj = {}
        config_obj['model'] = model
        config_obj['config']= config
        configs.append(config_obj)
    return configs


if __name__ == '__main__':
    # import ipdb as pdb; pdb.set_trace()
    args = parse_args()
    config_files = args['configs']
    base_em_dir  = args['base_nih_dir']
    seed         = args['seed']
    model_type = "UNET"
    configs = parse_configs(config_files, base_em_dir)
    var_input, var_gt = get_input_data()
    for config in configs:
        print("Running {0} model ...".format(config['config']['experiment_name'] ))
        cfg    = config['config']
        model  = config['model']
        if args['export_onnx']:
            # import ipdb as pdb; pdb.set_trace()
            batch, chnl, w, h = var_input.shape
            model.cpu()
            utility.export_torch_to_onnx(model, batch, chnl, w, h)
        else:
            get_result(var_input, var_gt, model, cfg, seed)


