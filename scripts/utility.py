import argparse
import os
import sys
import imageio
import torchvision.datasets as dsets
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path
import datetime
import numpy as np
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False, default="config.yaml")
    parser.add_argument('-t', '--modelype', help='type of model to run', required=False, default="UNET")
    args = parser.parse_args()
    return vars(args)

def remove_speciale_chars(str, chars = [",", ":", "-", " ", "."]):
    for char in chars:
        str = str.replace(char, "_")
    return str

def load_dataset(config):
    if config['dataset'] == 'mnist':
        # import ipdb as pdb; pdb.set_trace()
        # MNIST Dataset
        train_dataset = dsets.MNIST(root='./data/',
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)

        test_dataset = dsets.MNIST(root='./data/',
                                   train=False, 
                                   transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config["batchsize"], 
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config["batchsize"], 
                                                  shuffle=False)
    else: 
        print ("Unsupported dataset type".format(config['dataset']))
        sys.exit()
    return train_loader, test_loader, train_dataset, test_dataset

def save_model(model, config):
    experiment_name = get_experiment_name(config=config)
    experiment_name += ".pkl"
    experiment_dir = get_experiment_dir(config=config)
    model_file = experiment_dir + "/" + experiment_name
    if os.path.exists(model_file):
        current_datetime = str(datetime.datetime.now())
        current_datetime = remove_speciale_chars(current_datetime)
        copyfile(model_file, model_file + "_" + current_datetime)
    torch.save(model.state_dict(), model_file)
    print("Saving Model to {0}".format(model_file))

def get_experiment_name(config):
    if not ("experiment_name" in config):
        experiment_name =  None
    elif config["experiment_name"] == "":
        experiment_name = None
    else:
        experiment_name = config["experiment_name"]
    current_datetime = str(datetime.datetime.now())
    current_datetime = remove_speciale_chars(current_datetime)
    if experiment_name == None:
        experiment_name = current_datetime
    else:
        if os.path.exists(experiment_name+".pkl"):
            experiment_name += "_"+current_datetime
    return experiment_name

def save_image_to_path(img, image_name, path):
    # import ipdb as pdb; pdb.set_trace()
    if not os.path.exists(path):
        print ("Path {0} does not exist".format(path))
        try:  
            os.mkdir(path)
        except OSError:  
            print ("Creation of the directory {0} failed".format(path))
        else:  
            print ("Successfully created the directory {0}".format(path))
    path = path+image_name
    img = np.asarray( img, dtype="uint8" )
    imageio.imsave(path, img, "PNG")

def get_experiment_dir(config):
    experiment_name = get_experiment_name(config)
    if not("log_output_dir" in config):
        log_output_dir = "./"
    else:
        log_output_dir = config["log_output_dir"]
    if log_output_dir[-1] != "/":
        log_output_dir += "/"
    log_output_dir += experiment_name
    return log_output_dir

def create_log_file(config):
    # import ipdb as pdb; pdb.set_trace()
    expr_dir=get_experiment_dir(config)
    if not os.path.isdir(expr_dir):
        os.makedirs (expr_dir)
    log_file = expr_dir + "/" + get_experiment_name(config) + ".txt"
    if os.path.exists(log_file):
        current_datetime = str(datetime.datetime.now())
        current_datetime = remove_speciale_chars(current_datetime)
        copyfile(log_file, log_file + "_" + current_datetime)
    open(log_file, "w+").close()

def log_info(config, log_str):
#    import ipdb as pdb; pdb.set_trace()
    expr_dir=get_experiment_dir(config)
    log_file = expr_dir + "/" + get_experiment_name(config) + ".txt"
    with open(log_file, "a+") as f:
        f.write("{0}\n".format(log_str))
        f.close()

def export_torch_to_onnx(model, batch_size, nb_channels, w, h):
    import torch
    # import ipdb as pdb; pdb.set_trace()
    if isinstance(model, torch.nn.Module):
        model_name =  model.__class__.__name__
        # create the imput placeholder for the model
        # note: we have to specify the size of a batch of input images
        input_placeholder = torch.randn(batch_size, nb_channels, w, h)
        onnx_model_fname = model_name + ".onnx"
        # export pytorch model to onnx
        torch.onnx.export(model, input_placeholder, onnx_model_fname)
        print("{0} was exported to onnx: {1}".format(model_name, onnx_model_fname))
        return onnx_model_fname
    else:
        print("Unsupported model file")
        return
