import numpy as np
import utility
import torch
import sys
sys.path.insert(0, '../UNET_EM_DATASET/UNET_EM_DATASET_BASE/')
import models as m
from torchsummary import summary
import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='config file', required=True)
    args = parser.parse_args()
    return vars(args)

def get_summary(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg = models.vgg16().to(device)
    unet_model = m.Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    unet_model = unet_model.to(device)
    # import ipdb as pdb; pdb.set_trace()
    summary(unet_model, ( 1, 200, 200))

if __name__ == '__main__':
    # import ipdb as pdb; pdb.set_trace()
    args = parse_args()
    config_file = args['config_file']
    model_type = "UNET"
    config = config.Configuration(model_type, config_file)
    config = config.config_dict
    get_summary(config)

