import sys
sys.path.insert(0, '../UNET_GM_DATASET/UNET_Quant_Base/')
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

rand_seed = 0#randint(0, 100)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_gm_dir', help='path to gm experiment folder', required=False, default="/users/hemmat/MyRepos/UNET_Quantized/UNET_GM_DATASET/")
    parser.add_argument('-d', '--dataset_path', help='path to the GM dataset', required=False, default="/export/tmp/hemmat/datasets/medicaltorch/data/")
    # parser.add_argument('-c', '--config_file', help='config file', required=True)
    parser.add_argument('-c','--configs', nargs='+', help='configurations to be used', required=True)
    args = parser.parse_args()
    return vars(args)

def get_summary(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # import ipdb as pdb; pdb.set_trace()
    summary(unet_model, ( 1, 200, 200))

def get_input_data(dataset_path):
    val_transform = transforms.Compose([
        CenterCrop2D((200, 200)),
        ToTensor(),
        NormalizeInstance(),
    ])
    # Here we assume that the SC GM Challenge data is inside the folder
    # "../data" and it was previously resampled.
    gmdataset_val = SCGMChallenge2DTrain(root_dir=dataset_path,
                                                   subj_ids=range(9, 11),
                                                   transform=val_transform)

    val_loader = DataLoader(gmdataset_val, batch_size=1,
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_collate,
                            num_workers=1)
    # import ipdb as pdb; pdb.set_trace()
    for i, batch in enumerate(val_loader):
        input_samples, gt_samples = batch["input"], batch["gt"]
        var_input = input_samples.cuda()
        var_gt = gt_samples.cuda()
        break
    return var_input, var_gt

def get_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    unet_model = unet_model.to(device)
    unet_model.load_state_dict(torch.load(config['trained_model']))
    return unet_model

def normalize_img(img):
    # img = img.detach().cpu().numpy()
    img = img - img.min()
    img = img/img.max()
    img = img * 255
    return img


def get_result(var_input, var_gt, model, config):
    # import ipdb as pdb; pdb.set_trace()
    model = model.cpu()
    var_input = var_input.cpu()
    preds = model(var_input)
    save_image(var_input, var_gt, preds, "./", config) 
    return

def create_mask(img, mask_val, back_color):
    new_img = np.zeros(img.shape) 
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if np.abs(pixel - back_color) > 10:
                new_img[i][j] = mask_val
            else:
                new_img[i][j] = back_color
    return new_img

def stack_img(img, num_stacks=1):
    img     = img.detach().cpu().numpy()
    size    = img.shape
    steps   = size[-1]
    out_img = np.zeros((steps, steps*num_stacks))
    for i in range(0,num_stacks):
        out_img[:, i       : (i+1)*steps] = img[i][0]
    return out_img

def save_image(in_img, gt_img, pred_img, path, config):
    # import ipdb as pdb; pdb.set_trace()
    # in_img   = stack_img(in_img)
    # gt_img   = stack_img(gt_img)
    # pred_img = stack_img(pred_img)

    in_img   = normalize_img(in_img[0, 0,:,:]).cpu()
    gt_img   = normalize_img(gt_img[0, 0,:,:]).cpu()
    pred_img = normalize_img(pred_img[0, 0,:,:]).detach().cpu()
    img_name      = "GM_UNET_"      + config['experiment_name'] + "_" + str(rand_seed) + ".png"
    orig_img_name = "GM_UNET_ORIG_" + config['experiment_name'] + "_" + str(rand_seed) + ".png"
    gt_img_name   = "GM_UNET_GT_"   + config['experiment_name'] + "_" + str(rand_seed) + ".png"
    pred_img_name = "GM_UNET_PRED_" + config['experiment_name'] + "_" + str(rand_seed) + ".png"
    gt_img = create_mask(gt_img, 255, 0)
    pred_img = create_mask(pred_img, 255, 0)
    # utility.save_image_to_path(in_img, orig_img_name, path)
    # utility.save_image_to_path(gt_img, gt_img_name, path)
    # utility.save_image_to_path(pred_img, pred_img_name, path)
    # gt_over_layed   = lay.lay_over(gt_img,   in_img,        lay.yellow, alpha=0.5, mask_val=0)
    # pred_over_layed = lay.lay_over(pred_img, gt_over_layed, lay.red,    alpha=0.5, mask_val=0)
    pred_over_layed = lay.combine(gt_img, pred_img, lay.red, lay.blue, lay.orange, in_img, lay.black)
    utility.save_image_to_path(pred_over_layed, img_name, path)


if __name__ == '__main__':
    # import ipdb as pdb; pdb.set_trace()
    args = parse_args()
    config_files = args['configs']
    dataset_path = args['dataset_path']
    base_gm_dir  = args['base_gm_dir']
    model_type = "UNET"
    var_input, var_gt = get_input_data(dataset_path)
    for config_file in config_files:
        # import ipdb as pdb; pdb.set_trace()
        config_file = base_gm_dir+config_file+"/config.yaml"
        config = configuration.Configuration(model_type, config_file)
        config = config.config_dict
        model  = get_model(config)
        get_result(var_input, var_gt, model, config)


