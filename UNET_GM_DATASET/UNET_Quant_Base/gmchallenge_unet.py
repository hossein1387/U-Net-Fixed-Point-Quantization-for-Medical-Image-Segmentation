from collections import defaultdict
import time
import os

import numpy as np

from tqdm import tqdm

from tensorboardX import SummaryWriter

from datasets import *
from models import *
from transforms import *
from losses import *
from metrics import *
from filters import *
import model_visualizer as v
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torchvision.utils as vutils

import utility
import config

cudnn.benchmark = True
cuda = torch.cuda.is_available()

# Set seed for data random generator 
torch.manual_seed(0)
torch.cuda.manual_seed_all

def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def run_main(config):
    train_loss_total_avg = 0.0
    train_transform = transforms.Compose([
        CenterCrop2D((200, 200)),
        ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        RandomTensorChannelShift((-0.10, 0.10)),
        ToTensor(),
        NormalizeInstance(),
    ])

    val_transform = transforms.Compose([
        CenterCrop2D((200, 200)),
        ToTensor(),
        NormalizeInstance(),
    ])

#    import ipdb as pdb; pdb.set_trace()
   
    # Here we assume that the SC GM Challenge data is inside the folder
    # "data" and it was previously resampled.
    gmdataset_train = SCGMChallenge2DTrain(root_dir="data",
                                                    subj_ids=range(1, 9),
                                                    transform=train_transform,
                                                    slice_filter_fn=SliceFilter())

    # Here we assume that the SC GM Challenge data is inside the folder
    # "../data" and it was previously resampled.
    gmdataset_val = SCGMChallenge2DTrain(root_dir="data",
                                                   subj_ids=range(9, 11),
                                                   transform=val_transform)

    train_loader = DataLoader(gmdataset_train, batch_size=16,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_collate,
                              num_workers=1)

    val_loader = DataLoader(gmdataset_val, batch_size=16,
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_collate,
                            num_workers=1)

    # import ipdb as pdb; pdb.set_trace()
    
    utility.create_log_file(config)
    utility.log_info(config, "{0}\nStarting experiment {1}\n{0}\n".format(50*"=", utility.get_experiment_name(config)))
    model = Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    # print(model)
    #summary(model, (3, 224, 224))

    # import ipdb as pdb; pdb.set_trace()
    if config['operation_mode'].lower() == "retrain" or config['operation_mode'].lower() == "inference":
        print("Using a trained model...")
        model.load_state_dict(torch.load(config['trained_model']))
    elif config["operation_mode"].lower() == "visualize":
        print("Visualizing weights...")
        if cuda:
            model.load_state_dict(torch.load(config['trained_model']))
        else:
            model.load_state_dict(torch.load(config['trained_model'], map_location='cpu'))
        v.visualize_model(model, config)
        return 

    # import ipdb as pdb; pdb.set_trace()
    if cuda:
        model.cuda()

    num_epochs = config["num_epochs"]
    initial_lr = config["lr"]

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    betas = torch.linspace(3.0, 8.0, num_epochs)
    best_dice = 0
    # import ipdb as pdb; pdb.set_trace()
    writer = SummaryWriter(log_dir=utility.get_experiment_dir(config))
    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()

        if not(config['operation_mode'].lower() == "inference"):
            scheduler.step()

            lr = scheduler.get_lr()[0]
            model.beta = betas[epoch-1] # for ternary net, set beta
            writer.add_scalar('learning_rate', lr, epoch)

            model.train()
            train_loss_total = 0.0
            num_steps = 0
            for i, batch in enumerate(train_loader):
                input_samples, gt_samples = batch["input"], batch["gt"]
                if cuda:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda()
                else:
                    var_input = input_samples
                    var_gt = gt_samples
                preds = model(var_input)

                loss = dice_loss(preds, var_gt)
                # if epoch == 1 and i == len(train_loader) - 1:
                #     import ipdb as pdb; pdb.set_trace()
                # if epoch == 4 and i == len(train_loader) - 1:
                #     import ipdb as pdb; pdb.set_trace()
                train_loss_total += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_steps += 1

                if epoch % 5 == 0:
                    grid_img = vutils.make_grid(input_samples,
                                                normalize=True,
                                                scale_each=True)
                    writer.add_image('Input', grid_img, epoch)

                    grid_img = vutils.make_grid(preds.data.cpu(),
                                                normalize=True,
                                                scale_each=True)
                    writer.add_image('Predictions', grid_img, epoch)

                    grid_img = vutils.make_grid(gt_samples,
                                                normalize=True,
                                                scale_each=True)
                    writer.add_image('Ground Truth', grid_img, epoch)

        if not(config['operation_mode'].lower() == "inference"):
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
            # import ipdb as pdb; pdb.set_trace()
            input_samples, gt_samples = batch["input"], batch["gt"]

            with torch.no_grad():
                if cuda:
                    var_input = input_samples.cuda()
                    var_gt = gt_samples.cuda()
                else:
                    var_input = input_samples
                    var_gt = gt_samples

                preds = model(var_input)
                loss = dice_loss(preds, var_gt)
                val_loss_total += loss.item()

            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)
            gt_npy = gt_npy.squeeze(axis=1)

            preds = preds.data.cpu().numpy()
            # if np.isnan(preds).any(): 
            #     import ipdb as pdb; pdb.set_trace()
            preds = threshold_predictions(preds)
            preds = preds.astype(np.uint8)
            preds = preds.squeeze(axis=1)

            metric_mgr(preds, gt_npy)

            num_steps += 1
        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('metrics', metrics_dict, epoch)

        val_loss_total_avg = val_loss_total / num_steps

        if not(config['operation_mode'].lower() == "inference"):
            writer.add_scalars('losses', {
                                    'train_loss': train_loss_total_avg
                                }, epoch)
            writer.add_scalars('losses', {
                                    'val_loss': val_loss_total_avg,
                                    'train_loss': train_loss_total_avg
                                }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        log_str = "Epoch {} took {:.2f} seconds dice_score={}.".format(epoch, total_time, metrics_dict["dice_score"])
        utility.log_info(config, log_str)
        tqdm.write(log_str)
        if metrics_dict["dice_score"] > best_dice:
            best_dice = metrics_dict["dice_score"]
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
    #import ipdb as pdb; pdb.set_trace()
    if cuda:
        if "gpu_core_num" in config:
            device_id = int(config["gpu_core_num"])
            torch.cuda.set_device(device_id)
    run_main(config)
