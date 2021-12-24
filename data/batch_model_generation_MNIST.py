#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import os
import argparse
import argunparse
import subprocess

if __name__=='__main__':

    # Specify data generation configuration
    TOP_DIR: str =  "./data"                    # Top level directory that is used to hold all Trojaned models
    CLEAN_DATA_DIR: str = "./data/clean_data"   # Folder that holds the clean input images. Put your clean MNIST dataset here
    LOG_FILE: str = "./data/log"                # Log file name
    TENSORBOARD_DIR: str = "./data/tensorboard" # Tensorboard directory

    parser = argparse.ArgumentParser(description='MNIST Data Generation and Model Training')
    unparser=argunparse.ArgumentUnparser()
    parser.add_argument('--console', action='store_true')
    parser.add_argument('--tensorboard_dir', type=str,  help='Folder for logging tensorboard')
    parser.add_argument('--gpu', action="store_true", help="True to use GPU")
    parser.add_argument('--gpu_ind', type=str, default='0', help='Indices of GPUs to be used')
    parser.add_argument('--parallel',  action='store_true', help='Enable training with parallel processing, including multiple GPUs if available')
    parser.add_argument('--num_models', type=int, help='Number of models to be generated', default=1)
    # Use early stop to fulfill Trojan requirement, algorithm will run until stopping
    parser.add_argument('--num_epochs', type=int, help="Epoch before launching early stopping", default=20)
    parser.add_argument('--troj_frac', type=float, help='Fraction of target class images that are triggered', default=0.2)
    parser.add_argument('--target_class', type=str, help='Classes to be flipped to', default='0')
    parser.add_argument('--network', type=str, help='Experiment model architecture', default='leenet5', choices={'leenet5','resnet18'})
    a = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu_ind

    os.makedirs(TOP_DIR, exist_ok=True)
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    processes=[]
    for i in range(a.num_models):
        model_name="id-"+str(i).zfill(8)
        model_folder=os.path.join(TOP_DIR, model_name)
        if os.path.exists(model_folder):
            continue

        a.experiment_path = os.path.abspath(os.path.join(model_folder))                         # Top level dir
        a.models_output = os.path.abspath(os.path.join(model_folder))                           # DIR to hold model's .pt file
        a.log = os.path.abspath(LOG_FILE)                                                       # Log file path
        a.train = os.path.abspath(os.path.join(TOP_DIR, model_name, '/data/clean/train.csv'))   # Folder contains experiment training set
        a.test = os.path.abspath(os.path.join(TOP_DIR, model_name, '/data/clean/test.csv'))     # Folder contains experiment testing set
        a.train_experiment_csv = os.path.abspath(os.path.join(TOP_DIR, model_name, 'mnist_clean/train_mnist.csv'))
        a.test_experiment_csv = os.path.abspath(os.path.join(TOP_DIR,  model_name, 'mnist_clean/test_mnist.csv'))
        a.models_output = os.path.abspath(os.path.join(TOP_DIR, model_name))

        kwargs=a._get_kwargs()
        prefix='python gen_and_train_MNIST.py '
        arg_string=' '.join(['--'+str(x[0])+'='+str(x[1]) for x in kwargs[1:] if not (str(x[0]) in ['gpu', 'parallel'])])
        if ('gpu' in [str(x[0]) for x in kwargs]):
            arg_string+=' --gpu'
        if ('parallel' in [str(x[0]) for x in kwargs]):
            arg_string+=' --parallel'
        # cmd=prefix+arg_string+'&'  # Uncomment to send process to background and train all you model in parallel
        cmd=prefix+arg_string
        print(cmd)
        subprocess.run(cmd, shell=True)
