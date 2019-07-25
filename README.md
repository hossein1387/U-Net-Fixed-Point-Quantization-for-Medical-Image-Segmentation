# Fixed Point U-Net Quantization For Medical Image Segmentation
This repository contains code for "Fixed-Point U-Net Quantization for Medical Image Segmentation" paper to be appeared at MICCAI2019. It contains our experiments on three different datasets namely: the Spinal Cord Gray Matter Segmentation (GM), the ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM) and the public National Institute of Health (NIH) dataset for pancreas segmentation in abdominal CT scans.


## Data preprocessing:

For each dataset, we used a preprocessing script that can be found in . Please follow instructions for each dataset.
You can also download the pre-processed data from [this link]{https://drive.google.com/file/d/1kjc3HLVuGdMa9wBF1SHaNicH9Y-maDzZ/view?usp=sharing}

## Running code:

Every dataset contains a main directory called ***_BASE. This directory contains the original code for that dataset. The files found in folders in the dataset directory are symbolically linked to the files in BASE directory except the config file. The configuration file is a YAML file that shows what configuration is used for this specific experiment. For instance, for EM dataset, to run an experiment with a specific integer quantization precision (lets try Q4.4 bit for weight and Q4.4 bit for activation), you first need to modify the configuration as follow:

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
