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
# torch.set_printoptions(threshold=10000)
# def _to_fixed_point(v, i, f):
#     # import ipdb as pdb; pdb.set_trace()
#     pows = torch.arange(-f, 0, 1)
#     max_float = Variable(torch.zeros(v.size()).type(v.data.type()) + torch.pow(2*torch.ones(pows.size()), pows.float()).sum()) if f!=0 else Variable(torch.zeros(v.size()).type(v.data.type()))
#     max_int   = Variable(torch.zeros(v.size()).type(v.data.type()) + 2**(i-1)) if i!=0 else Variable(torch.zeros(v.size()).type(v.data.type()))
#     max_ =    torch.ones(v.size())* (max_int + max_float)
#     min_ = -1*torch.ones(v.size())* (max_int + max_float)
#     max_mask = v>max_
#     v = max_mask.float()*max_ + (1-max_mask.float())*v
#     min_mask = v<min_
#     v = min_mask.float()*min_ + (1-min_mask.float())*v
#     return v

# def to_fixed_point(v, i, f):
#     # Convert to fixed point : round(abs(value) * pow(2, frac))
#     #import ipdb as pdb; pdb.set_trace()
#     if isinstance(v, Variable):
#         max_ = Variable(torch.zeros(v.size()).type(v.data.type()) + ((1 << (i + f)) - 1))
#         min_ = Variable(torch.zeros(v.size()).type(v.data.type()) + (-((1 << (i+f)) - 1)))
#     else:
#         max_ = torch.zeros(v.size()).type(v.type()) + ((1 << (i + f)) - 1)
#         min_ = torch.zeros(v.size()).type(v.type()) + (-((1 << (i + f)) - 1))
#     v = torch.max(min_, torch.min(torch.sign(v) * torch.round(torch.abs(v) * 2**f), max_)) # two complement to check the min (ask hossein)
#     return v

def plot_tensor(tensor, ax, name, config):
    tensor = torch.abs(tensor)
    #import ipdb as pdb; pdb.set_trace()
    #print(tensor.shape)
    ax.set_title(name, fontsize="small")
    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=6)
    if config["quantization"].lower() == "fixed":
        #ax.hist(quant.to_fixed_point(tensor, config["weight_i_width"], config["weight_f_width"]).detach().numpy(), normed=True)
        sns.distplot(quant.to_fixed_point(tensor, config["weight_i_width"], config["weight_f_width"]).detach().numpy(), ax=ax, kde_kws={"color": "r"})
        #sns.kdeplot(quant.to_fixed_point(tensor, config["weight_i_width"], config["weight_f_width"]).detach().numpy(), ax=ax,  shade=True)
    elif config["quantization"].lower() == "normal":
        sns.distplot(tensor.detach().numpy(), ax=ax, kde_kws={"color": "r"})

def visualize_model(model, config):
    plt.gcf().clear()
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3)
    # import ipdb as pdb; pdb.set_trace()
    unet_conv_1  = model.conv1.conv1.weight.view(-1)
    plot_tensor(unet_conv_1, ax1, "unet_conv1.weight", config)
    unet_conv_2  = model.conv1.conv2.weight.view(-1)
    plot_tensor(unet_conv_2, ax2, "unet_conv2.weight", config)
    unet_conv_3  = model.conv2.conv1.weight.view(-1)
    plot_tensor(unet_conv_3, ax3, "unet_conv3.weight", config)
    unet_conv_4  = model.conv2.conv2.weight.view(-1)
    plot_tensor(unet_conv_4, ax4, "unet_conv4.weight", config)
    unet_conv_5  = model.conv3.conv1.weight.view(-1)
    plot_tensor(unet_conv_5, ax5, "unet_conv5.weight", config)
    unet_conv_6  = model.conv3.conv2.weight.view(-1)
    plot_tensor(unet_conv_6, ax6, "unet_conv6.weight", config)
    unet_conv_7  = model.conv4.conv1.weight.view(-1)
    plot_tensor(unet_conv_7, ax7, "unet_conv7.weight", config)
    unet_conv_8  = model.conv4.conv2.weight.view(-1)
    plot_tensor(unet_conv_8, ax8, "unet_conv8.weight", config)

    unet_conv_9  = model.conv9.weight.view(-1)
    plot_tensor(unet_conv_9, ax9, "unet_conv9.weight", config)

    unet_conv_10 = model.up1.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_10, ax10, "unet_conv10.weight", config)
    unet_conv_11 = model.up1.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_11, ax11, "unet_conv11.weight", config)
    unet_conv_12 = model.up2.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_12, ax12, "unet_conv12.weight", config)
    unet_conv_13 = model.up2.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_13, ax13, "unet_conv13.weight", config)
    unet_conv_14 = model.up3.downconv.conv1.weight.view(-1)
    plot_tensor(unet_conv_14, ax14, "unet_conv14.weight", config)
    unet_conv_15 = model.up3.downconv.conv2.weight.view(-1)
    plot_tensor(unet_conv_15, ax15, "unet_conv15.weight", config)
    
    fig.tight_layout()
    # st = fig.suptitle(config["experiment_name"], fontsize="x-large")
    # st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig(config["experiment_name"]+".pdf", format="pdf")


# Using smooth function to smooth the data
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_dice_score_curve(config):
    # import ipdb as pdb; pdb.set_trace()
    plt.gcf().clear()
    scores = []
    expr_dir = utility.get_experiment_dir(config)
    log_file = expr_dir + "/" + utility.get_experiment_name(config) + ".txt"
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "dice_score" in line:
                score = float(line.split("dice_score=")[1].replace("\n","")[:-1])
                scores.append(score)
    scores_smooth = smooth(scores, 20)
    scores_smooth[-20:] = scores[-20:]
    plt.plot(scores_smooth)
    plt.savefig(config["experiment_name"] + "dice_score" + ".pdf")




if __name__ == '__main__':
    # import ipdb as pdb; pdb.set_trace()
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    expr_dir = utility.get_experiment_dir(config)
    model_file = expr_dir + "/" + utility.get_experiment_name(config) + ".pkl"
    model = Unet(drop_rate=0.4, bn_momentum=0.1, config=config)
    model.load_state_dict(torch.load(model_file))
    visualize_model(model, config)
    plot_dice_score_curve(config)
