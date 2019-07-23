import numpy            as np
import torch
import torch.nn as nn
from torch.nn import (Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, ReLU, BatchNorm1d, Linear, CrossEntropyLoss)
from torch.nn import Module
import torch.nn.functional as F
from   layers       import *

def ternaryTanh(x,beta=2.0):
    m = torch.nn.Tanh()
    if(beta>=1.0):
        y = m((x*beta*2.0-beta))*0.5
        y += -m((-x*beta*2.0-beta))*0.5
    elif(beta==0.0):
        y = torch.sign(x)
    elif(beta<0):
        y = torch.nn.HardTanh(x)
    else:
        y = torch.sign(x)*((torch.abs(x)>beta).float())
    return y

class DownConv(Module):
    def __init__(self, in_feat, out_feat, config, drop_rate=0.4, bn_momentum=0.1, firstlayer=False):
        super(DownConv, self).__init__()
        self.firstlayer = firstlayer
        self.config = config
        # import  ipdb as pdb; pdb.set_trace()
        if config['quantization'].lower() != "normal" and not firstlayer:
            self.conv1 = Conv2dQuant(in_feat , out_feat, kernel_size=(3,3), padding=1, H=1, W_LR_scale="Glorot", config=config, stride=1)
            self.conv2 = Conv2dQuant(out_feat, out_feat, kernel_size=(3,3), padding=1, H=1, W_LR_scale="Glorot", config=config, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_feat , out_feat, kernel_size=3, padding=1, stride=1)
            self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, stride=1)

        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        # self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        # self.conv2_drop = nn.Dropout2d(drop_rate)
        self.quant = QuantLayer(config=config)
        
    def forward(self, x, beta=2.0):
        # import ipdb as pdb; pdb.set_trace()
        x = self.conv1(x)
        if self.config['activation'].lower()=="ternarytanh" and not self.firstlayer:
            x = ternaryTanh(x, beta)
        else:
            x = F.relu(x)
        x = self.conv1_bn(x)
        # x = self.conv1_drop(x)
        x = self.quant(x)

        x = self.conv2(x)
        if self.config['activation'].lower()=="ternarytanh" and not self.firstlayer:
            x = ternaryTanh(x, beta)
        else:
            x = F.relu(x)
        x = self.conv2_bn(x)
        x = self.quant(x)
        # x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, config, drop_rate=0.4, bn_momentum=0.1, beta=2.0):
        super(UpConv, self).__init__()
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up1 = nn.functional.interpolate(scale_factor=2, mode='nearest', align_corners=False)
        self.downconv = DownConv(in_feat=in_feat, out_feat=out_feat, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)

    def forward(self, x, y, beta):
        # x = self.up1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x, beta)
        return x


class Unet(Module):
    """A reference U-Net model.
    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, config, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()
        # import  ipdb as pdb; pdb.set_trace()
        self.beta = 3.0
        #Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config, firstlayer=True)
        #self.conv1 = DownConv(1, 64, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config, firstlayer=False)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)
        self.up2 = UpConv(384, 128, drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)
        self.up3 = UpConv(192, 64 , drop_rate=drop_rate, bn_momentum=bn_momentum, config=config)

        # if config['quantization'].lower() != "normal":
        #     self.conv9 = Conv2dQuant(64, 1, kernel_size=(3,3), padding=1, H=1, W_LR_scale="Glorot", config=config)
        # else:
        #     self.conv9 = nn.Conv2d(64, 1, kernel_size=(3,3), padding=1)
        #self.conv9 = Conv2dQuant(64, 1, kernel_size=(3,3), padding=1, H=1, W_LR_scale="Glorot", config=config)
        self.conv9 = nn.Conv2d(64, 1, kernel_size=(3,3), padding=1)

    def forward(self, x):
        # import  ipdb as pdb; pdb.set_trace()
        x1 = self.conv1(x, self.beta )
        x2 = self.mp1(x1)

        x3 = self.conv2(x2, self.beta )
        x4 = self.mp2(x3)

        x5 = self.conv3(x4, self.beta )
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6, self.beta )

        # Up-sampling
        x8  = self.up1(x7, x5, self.beta )
        x9  = self.up2(x8, x3, self.beta )
        x10 = self.up3(x9, x1, self.beta )

        x11 = self.conv9(x10 )
        # preds = F.sigmoid(x11)
        preds = torch.sigmoid(x11)

        return preds
