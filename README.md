![alt text](https://github.com/hossein1387/Fixed-Point-U-Net-Quantization-for-Medical-Image-Segmentation/blob/master/Figs/nih.png)
![alt text](https://github.com/hossein1387/Fixed-Point-U-Net-Quantization-for-Medical-Image-Segmentation/blob/master/Figs/em.png)
![alt text](https://github.com/hossein1387/Fixed-Point-U-Net-Quantization-for-Medical-Image-Segmentation/blob/master/Figs/gm.png)


#  [U-Net Fixed Point Quantization For Medical Image Segmentation](https://arxiv.org/abs/1908.01073) ![]( https://visitor-badge.glitch.me/badge?page_id=hossein1387.U-Net-Fixed-Point-Quantization-for-Medical-Image-Segmentation)

This repository contains code for "U-Net Fixed-Point Quantization for Medical Image Segmentation
" paper to be appeared at MICCAI2019. It contains our experiments on three different datasets namely: [The Spinal Cord Gray Matter Segmentation (GM)](https://www.sciencedirect.com/science/article/pii/S1053811917302185), [The ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000502) and [The public National Institute of Health (NIH) dataset for pancreas segmentation in abdominal CT scans](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT).


## Data pre-processing:

For each dataset, we used a pre-processing script that can be found in [pre-processing](https://github.com/hossein1387/U-Net-Fixed-Point-Quantization-for-Medical-Image-Segmentation/tree/master/preprocessing) directory. Please follow instructions for each dataset. For GM, there is no seperate pre-processing script. Pre-processing happens automatically before training. 
You can also download the pre-processed data from [this link](https://drive.google.com/file/d/1kjc3HLVuGdMa9wBF1SHaNicH9Y-maDzZ/view?usp=sharing).

## Configuring the Model using config.yaml:

Every dataset contains a main directory called \*\*\*\_BASE. This directory contains the original code for that dataset. The files found in folders in the dataset directory are symbolically linked to the files in BASE directory except the config file. The configuration file is a YAML file that shows what configuration is used for this specific experiment. For instance, for EM dataset, to run an experiment with a specific integer quantization precision (lets try Q4.4 bit for weight and Q4.4 bit for activation), you first need to modify the configuration as follow:

```yaml

UNET:
    dataset: 'emdataset'
    lr: 0.001
    num_epochs: 200
    model_type: "unet"
    init_type: glorot
    quantization: "FIXED"
    activation_f_width: 4
    activation_i_width: 4
    weight_f_width: 4
    weight_i_width: 4
    gpu_core_num: 1
    trained_model: "/path/to/trained/models/em_a4_4_w4_4.pkl"
    experiment_name: "em_a4_4_w4_4"
    log_output_dir: "/path/to/output/folder"
    operation_mode: "normal"
```

All datasets use the same configuration format. The following are most of the configuration that can be used:

* Currently we only have tested our quantizer on UNET architecture hence, the first line must be `UNET` and it must be passed as argument to the main script.
* `dataset`: Defines the dataset name. It can be named anything. The scripts use this string for outputting readable information. 
* `lr`: Shows an initial learning rate. We use Annealing Cosine scheduler for reducing the learning rate. 
* `num_epochs`: Defines how many epochs you wish to run your model.
* `model_type`: Is a string that shows what model we use, currently, the only option is `unet`. 
* `init_type`: Is the initialization function that is used for initializing parameters. Currently the only option is `glorot`.
* `quantization`: Defines what type of quantization you want to use. You can choose: 
    * `INT`: Integer quantization
    * `BNN`: Binary quantization
    * `Normal`: No quantization
    * `FIXED`: Fixed point quantization
    * `Ternary`: Ternary Quantization
* `activation_[f-i]_width`: is used to define how many bits you want to use for quantizing the floating (`f`) or integer (`i`) part of the activation values. This option is used only for `INT` and `FIXED` quantization types.
* `weight_[f-i]_width`: is used to define how many bits you want to use for quantizing the floating (`f`) or integer (`i`) part of the parameter values. This option is used only for `INT` and `FIXED` quantization types.
* `gpu_core_num`: Defines which gpu core you want to run your model. Parallel computing (multiple gpus) is not supported for the moment.
* `trained_model`: Path to save best model while training.
* `experiment_name`: A name for the experiment
* `log_output_dir`: Path to output log.
* `operation_mode`: Can be any of the following:
    * `normal`: the model will be put into training.
    * `visualize`: `trained_model` will be used top plot weight distribution of every layer.
    * `retrain`:  `trained_model` will be used as the initial state for training. 
    * `inference`: `trained_model` will be used to run one batch of data and the accuracy will be printed out.

## Running code:

After the configurations are set properly, you can run the following command to start the requested opration (the following shows command
to run an em dataset experiment):

`python em_unet.py -f config.yaml -t UNET`


## Citation

If you found our work interesting, please consider citing our paper:

MohammadHossein AskariHemmat, Sina Honari, Lucas Rouhier, Christian S. Perone, Julien Cohen-Adad, Yvon Savaria, Jean-Pierre David, U-Net Fixed-Point Quantization for Medical Image Segmentation, Hardware Aware Learning for
Medical Imaging and Computer Assisted Intervention (HAL-MICCAI), 2019. 

Bibtex:

    @inproceedings{askarimiccai2019,
    title={U-Net Fixed Point Quantization For Medical Image Segmentation},
    author={AskariHemmat, MohammadHossein and Honari, Sina and Rouhier, Lucas  and S. Perone, Christian  and Cohen-Adad, Julien and Savaria, Yvon and David, Jean-Pierre},
    booktitle={Medical Imaging and Computer Assisted Intervention (MICCAI), Hardware Aware Learning Workshop (HAL-MICCAI) 2019},
    year={2019},
    publisher={Springer}
    }
