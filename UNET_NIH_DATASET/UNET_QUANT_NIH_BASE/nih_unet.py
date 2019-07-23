from models import *
import torch
from data import *
import models as m
from datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import defaultdict
import time
import os
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from transforms import *
from losses import *
from metrics import *
from filters import *
import utility
import config
from natsort import natsorted
from glob import glob
from matplotlib.image import imread 
import layover as lay

cudnn.benchmark = True
cuda = torch.cuda.is_available()


def get_samples(image_val_path, target_val_path, num_samples=4):
    # import ipdb as pdb; pdb.set_trace()
    dim1, dim2 = imread(image_val_path[0]).shape
    image_val  = torch.ones([4, 1, dim1, dim2 ])
    target_val = torch.ones([4, 1, dim1, dim2 ])
    for i in range(0, num_samples):
        image_val [i,0,:,:] = torch.from_numpy(imread(image_val_path[i]))
        target_val[i,0,:,:] = torch.from_numpy(imread(target_val_path[i]))
    return image_val, target_val

def save_image(in_img, gt_img, pred_img, epoch, idx, path):
    # if epoch % 5 == 0 :
        in_img  *= 255
        pred_img*= 255
        gt_img  *= 255
        pred_img = pred_img.astype(int)
        # import ipdb as pdb; pdb.set_trace()
        # if epoch > 198:
        #     import ipdb as pdb; pdb.set_trace()
        img_name      = "NIH2D_UNET_"      + str(idx) + str(epoch) + ".png"
        orig_img_name = "NIH2D_UNET_ORIG_" + str(idx) + str(epoch) + ".png"
        gt_img_name   = "NIH2D_UNET_GT_"   + str(idx) + str(epoch) + ".png"
        pred_img_name = "NIH2D_UNET_PRED_" + str(idx) + str(epoch) + ".png"
        utility.save_image_to_path(in_img, orig_img_name, path)
        utility.save_image_to_path(gt_img, gt_img_name, path)
        utility.save_image_to_path(pred_img, pred_img_name, path)

        gt_over_layed   = lay.lay_over(gt_img,   in_img,        lay.yellow, alpha=0.2, mask_val=255)
        pred_over_layed = lay.lay_over(pred_img, gt_over_layed, lay.red,    alpha=0.2, mask_val=255)
        utility.save_image_to_path(pred_over_layed, img_name, path)

def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def run_main(config):
    dataset_base_path = "./data/"
    target_path       = natsorted(glob(dataset_base_path + '/mask/*.png'))
    image_paths       = natsorted(glob(dataset_base_path + '/img/*.png'))
    target_val_path   = natsorted(glob(dataset_base_path + '/val_mask/*.png'))
    image_val_path    = natsorted(glob(dataset_base_path + '/val_img/*.png'))

    nih_dataset_train = EMdataset(image_paths=image_paths, target_paths=target_path)
    nih_dataset_val = EMdataset(image_paths=image_val_path, target_paths=target_val_path)
     
    #import ipdb as pdb; pdb.set_trace()
    train_loader = DataLoader(nih_dataset_train, batch_size=16, shuffle=True, num_workers=1)
    val_loader = DataLoader(nih_dataset_val, batch_size=16, shuffle=True, num_workers=1)
    model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    if config['operation_mode'].lower() == "retrain" or config['operation_mode'].lower() == "inference":
        print("Using a trained model...")
        model.load_state_dict(torch.load(config['trained_model']))
    elif config["operation_mode"].lower() == "visualize":
        print("Using a trained model...")
        if cuda:
            model.load_state_dict(torch.load(config['trained_model']))
        else:
            model.load_state_dict(torch.load(config['trained_model'], map_location='cpu'))
        v.visualize_model(model, config)
        return 

    # import ipdb as pdb; pdb.set_trace()
    if cuda:
        model.cuda()
        print('gpu_activate')

    num_epochs      = config["num_epochs"]
    initial_lr      = config["lr"]
    experiment_path = config["log_output_dir"] + config['experiment_name']
    output_image_dir= experiment_path + "/figs/"

    betas = torch.linspace(3.0, 8.0, num_epochs)

    # criterion  = nn.BCELoss()
    optimizer  = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # import ipdb as pdb; pdb.set_trace()
    writer = SummaryWriter(log_dir=utility.get_experiment_dir(config))
    best_score = 0
    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        model.beta = betas[epoch-1] # for ternary net, set beta
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        capture =True
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch[0], batch[1]

            if cuda:
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda()
            else:
                var_input = input_samples
                var_gt = gt_samples
            preds = model(var_input)
            loss = dice_loss(preds, var_gt)
            # import ipdb as pdb; pdb.set_trace()
            var_gt=var_gt.float()
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1
            if epoch % 1 == 0 and capture:
                capture = False
                input_samples, gt_samples = get_samples(image_val_path, target_val_path, 4)
                if cuda: 
                    input_samples = input_samples.cuda()
                preds = model(input_samples)
                input_samples = input_samples.data.cpu().numpy()
                preds = preds.data.cpu().numpy()
                # import ipdb as pdb; pdb.set_trace()
                save_image(input_samples[0][0], gt_samples[0][0], preds[0][0], epoch, 0  , output_image_dir)

        train_loss_total_avg = train_loss_total / num_steps

        # import ipdb as pdb; pdb.set_trace()
        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        metric_fns = [dice_score,
                      hausdorff_score,
                      precision_score,
                      recall_score,
                      specificity_score,
                      intersection_over_union,
                      accuracy_score]

        metric_mgr = MetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda(async=True)
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                preds = model(var_input)
                loss = dice_loss(preds, var_gt)
                # loss = criterion(preds, var_gt)
                # loss = weighted_bce_loss(preds, var_gt, 0.5, 2.5)
                val_loss_total += loss.item()

            gt_npy = gt_samples.data.cpu().numpy()#.astype(np.uint8)
            gt_npy = gt_npy.squeeze(axis=1)

            preds = preds.data.cpu().numpy()
            preds = threshold_predictions(preds)
            # preds = preds.astype(np.uint8)
            preds = preds.squeeze(axis=1)

            metric_mgr(preds, gt_npy)

            num_steps += 1

        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('metrics', metrics_dict, epoch)

        val_loss_total_avg = val_loss_total / num_steps

        writer.add_scalars('losses', {
                                'val_loss': val_loss_total_avg,
                                'train_loss': train_loss_total_avg
                            }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        msg = "Epoch {} took {:.2f} seconds dice_score={}. precision={} iou={} loss_train={} val_loss={}".format(epoch, total_time, 
                                                                                                                 metrics_dict["dice_score"],
                                                                                                                 metrics_dict["precision_score"],
                                                                                                                 metrics_dict["intersection_over_union"],
                                                                                                                 train_loss_total_avg,val_loss_total_avg)
        utility.log_info(config, msg)
        tqdm.write(msg)
        writer.add_scalars('losses', { 'train_loss': train_loss_total_avg }, epoch)

        if metrics_dict["dice_score"] > best_score:
            best_score = metrics_dict["dice_score"]
            utility.save_model(model=model, config=config)
            
    if not (config['operation_mode'].lower() == "inference"):
        utility.save_model(model=model, config=config)

if __name__ == '__main__':
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    # import ipdb as pdb; pdb.set_trace()
    if cuda:
        if "gpu_core_num" in config:
            device_id = int(config["gpu_core_num"])
            torch.cuda.set_device(device_id)
    run_main(config)
