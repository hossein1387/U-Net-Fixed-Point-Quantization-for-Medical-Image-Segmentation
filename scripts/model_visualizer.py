import sys
sys.path.insert(0, '../UNET_GM_DATASET/UNET_Quant_Base/')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import torch 
from torch.autograd import Variable
import utility
import config
from models import *
import quantize            as quant
torch.set_printoptions(precision=10)

def plot_tensor(tensor, ax, name, config, plot_type="kde"):
    ax.set_title(name, fontsize="small")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    # import ipdb as pdb; pdb.set_trace()
    if config["quantization"].lower() == "fixed":
        i_w = config["weight_i_width"] 
        f_w = config["weight_f_width"] 
        if plot_type.lower()=="kde":
            sns.distplot(quant.to_fixed_point(quant.to_nearest_power_of_two(tensor.cpu()), i_w, f_w).cpu().detach().numpy(), ax=ax, kde_kws={"color": "r"})
        elif plot_type.lower()=="hist":
            sns.distplot(quant.to_fixed_point(quant.to_nearest_power_of_two(tensor.cpu()), i_w, f_w).cpu().detach().numpy(), ax=ax, kde=False, hist=True, rug=True, norm_hist=False, bins=100)
        else:
            print("Unsupported plot type")
    elif config["quantization"].lower() == "normal":
        if plot_type.lower()=="kde":
            sns.distplot(tensor.detach().numpy(), ax=ax, kde_kws={"color": "r"})
        elif plot_type.lower()=="hist":
            sns.distplot(tensor.detach().numpy(), ax=ax, kde=False, hist=True, rug=True, norm_hist=False, bins=100)
        else:
            print("Unsupported plot type")


def visualize_model(model, config):
    plt.gcf().clear()
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
#    import ipdb as pdb; pdb.set_trace()
    unet_conv_1  = model.conv1.conv1.weight.view(-1)
    plot_tensor(unet_conv_1, ax1, "unet_conv1.weight", config, plot_type="kde")
    unet_conv_2  = model.conv1.conv2.weight.view(-1)
    plot_tensor(unet_conv_2, ax2, "unet_conv2.weight", config, plot_type="kde")
    unet_conv_3  = model.conv2.conv1.weight.view(-1)
    plot_tensor(unet_conv_3, ax3, "unet_conv3.weight", config, plot_type="kde")
    unet_conv_4  = model.conv2.conv2.weight.view(-1)
    plot_tensor(unet_conv_4, ax4, "unet_conv4.weight", config, plot_type="kde")
    unet_conv_5  = model.conv3.conv1.weight.view(-1)
    plot_tensor(unet_conv_5, ax5, "unet_conv5.weight", config, plot_type="kde")
    unet_conv_6  = model.conv3.conv2.weight.view(-1)
    plot_tensor(unet_conv_6, ax6, "unet_conv6.weight", config, plot_type="kde")
    unet_conv_7  = model.conv4.conv1.weight.view(-1)
    plot_tensor(unet_conv_7, ax7, "unet_conv7.weight", config, plot_type="kde")
    unet_conv_8  = model.conv4.conv2.weight.view(-1)
    plot_tensor(unet_conv_8, ax8, "unet_conv8.weight", config, plot_type="kde")

    unet_conv_9  = model.conv9.weight.view(-1)
    plot_tensor(unet_conv_9, ax9, "unet_conv9.weight", config, plot_type="kde")

    unet_conv_10 = model.up1.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_10, ax10, "unet_conv10.weight", config, plot_type="kde")
    unet_conv_11 = model.up1.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_11, ax11, "unet_conv11.weight", config, plot_type="kde")
    unet_conv_12 = model.up2.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_12, ax12, "unet_conv12.weight", config, plot_type="kde")
    unet_conv_13 = model.up2.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_13, ax13, "unet_conv13.weight", config, plot_type="kde")
    unet_conv_14 = model.up3.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_14, ax14, "unet_conv14.weight", config, plot_type="kde")
    unet_conv_15 = model.up3.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_15, ax15, "unet_conv15.weight", config, plot_type="kde")
    
    fig.tight_layout()
    # st = fig.suptitle(config["experiment_name"], fontsize="x-large")
    # st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig(config["experiment_name"]+".pdf", format="pdf")


if __name__ == '__main__':
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    model = Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    model.load_state_dict(torch.load(config['trained_model']))
    visualize_model(model, config)
