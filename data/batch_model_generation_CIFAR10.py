#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import os
import argparse
import subprocess

# Generate Trojaned models
if __name__=='__main__':

    # Specify data generation configuration
    TOP_DIR: str =  "./data"                    # Top level directory that is used to hold all Trojaned models
    CLEAN_DATA_DIR: str = "./data/clean_data"   # Folder that holds the clean input images. Put your clean CIFAR10 dataset here
    LOG_FILE: str = "./data/log"                # Log file name
    TENSORBOARD_DIR: str = "./data/tensorboard" # Tensorboard directory

    parser = argparse.ArgumentParser(description='CIFAR10 Data & Model Generator and Experiment')
    parser.add_argument('--console', action='store_true', help="Write output to console")
    parser.add_argument('--train_val_split', help='Amount of train data to use for validation', default=0.1, type=float)
    parser.add_argument('--gpu', action='store_true', help="True to use GPU")
    parser.add_argument('--gpu_index', default='0', help='Index of GPUs to be used')
    parser.add_argument('--early_stopping', action='store_true', help="True to apply early stopping")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs to use for each model")
    parser.add_argument('--num_models', type=int, help='Epoch before launching early stopping', default=20)
    parser.add_argument('--troj_frac', type=float, default=0.2, help='Trojan fraction')
    parser.add_argument('--target_class', type=str, default=0, help='Target class to be flipped to')
    parser.add_argument('--network', type=str, help='Architecture to be used', default='resnet18', choices={'resnet18', 'densenet121'})
    a = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_index

    os.makedirs(TOP_DIR, exist_ok=True)
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    for i in range(a.num_models):
        model_name="id-"+str(i).zfill(8)
        model_folder=os.path.join(TOP_DIR, model_name)
        if os.path.exists(model_folder):
            continue

        a.experiment_path = os.path.abspath(os.path.join(CLEAN_DATA_DIR))           # Top level dir
        a.data_folder = os.path.abspath(os.path.join(TOP_DIR, model_name, 'data'))  # DIR to hold generated example data (trojaned and clean)
        a.models_output = os.path.abspath(os.path.join(model_folder))               # DIR to hold model's .pt file

        kwargs=a._get_kwargs()
        prefix='python gen_and_train_cifar10.py '
        arg_string=' '.join(['--'+str(x[0])+'='+str(x[1]) for x in kwargs[1:] if not (str(x[0]) in ['gpu', 'parallel', 'early_stopping'])])
        if ('gpu' in [str(x[0]) for x in kwargs]):
            arg_string+=' --gpu'
        if ('parallel' in [str(x[0]) for x in kwargs]):
            arg_string+=' --parallel'
        if ('early_stopping' in [str(x[0]) for x in kwargs]):
            arg_string+=' --early_stopping'
        # cmd=prefix+arg_string+'&'  # Uncomment to send process to background and train all you model in parallel
        cmd=prefix+arg_string
        print(cmd)
        subprocess.run(cmd, shell=True)