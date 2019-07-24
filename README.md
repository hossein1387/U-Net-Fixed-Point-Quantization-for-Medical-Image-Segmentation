# Fixed-Point-U-Net-Quantization-for-Medical-Image-Segmentation
This repository contains code for "Fixed-Point U-Net Quantization for Medical Image Segmentation" paper to be appeared at MICCAI2019. It contains our experiments on three different datasets namely: the Spinal Cord Gray Matter Segmentation (GM), the ISBI challenge for segmentation of neuronal structures in Electron Microscopic (EM) and the public National Institute of Health (NIH) dataset for pancreas segmentation in abdominal CT scans.

## Data preprocessing:

For each dataset, we used a preprocessing script that can be found in ***. Please follow instructions for each dataset.


## Running code:

Every dataset contains a main directory called ***_BASE. This directory contains the original code for that dataset. The files found in folders in the dataset directory are symbolically linked to the files in BASE directory except the config file. The configuration file is a YAML file that shows what configuration is used for this specific experiment
