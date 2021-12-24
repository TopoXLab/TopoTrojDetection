#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import os
import gc
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import skimage.io

# Total number of neurons to be sampled
SAMPLE_LIMIT = 1.5e3

def img_std(img):
    """
    Reshape and rescale the input images to fit the model.
    """
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy+224, dx:dx+224, :]
    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    # convert image to a gpu tensor
    batch_data = torch.FloatTensor(img);

    return batch_data


def parse_arch(model: torch.tensor)-> Tuple[List, List]:
    """
    Parse a input model to extact layer-wise (Conv2d or Linear) module and corresponding module name.
    Input args:
        model (torch.nn.Module): A torch network
    Return:
        layer_list (List): A list contain all Conv2d and Linear module from shallow to deep
        layer_k (List): A list contain names of extracted modules in layer_list
    """

    layer_list = []
    layer_k = []
    for k in model._modules:
        if model._modules[k]._modules:
            # If it has child module then recursively extract the child module
            sub_layer_list, sub_layer_k = parse_arch(model._modules[k])
            layer_list += sub_layer_list
            layer_k += [k+'_'+x for x in sub_layer_k]
        elif isinstance(model._modules[k], torch.nn.Conv2d) or isinstance(model._modules[k], torch.nn.Linear):
            layer_list.append(model._modules[k])
            layer_k.append(model._modules[k]._get_name())
    return layer_list, layer_k


def feature_collect(model: torch.tensor, images: torch.tensor)-> Tuple[Dict, torch.tensor]:
    """
    Helper function to collection intermediate output of a model for given inputs.
    Input args:
        model (torch.nn.Module): A torch network
        images (torch.tensor): A valid image torch.tensor
    Return:
         feature_dict (dict): A dictionary contain all intermediate output tensor whose key is the (layer depth, module name)
         output (torch.tensor): final output of model
    """
    outs = []
    # Hook function to be registered during the forward procedure to collect intermediate output
    def feature_hook(module, f_in, f_out):
        if isinstance(f_in, torch.Tensor):
            outs.append(f_in.detach().cpu())
        else:
            outs.append(f_in[0].detach().cpu())
    module_list, module_k = parse_arch(model)
    feature_dict = {}
    handle_list = []
    # Keep registration handle to remove registration later
    for layer_ind in range(len(module_list)):
        handle_list.append(module_list[layer_ind].register_forward_hook(hook=feature_hook))
    output = model(images)
    for layer_ind in range(len(module_list)):
        #  if layer_ind in layer_select:
        feature_dict[(layer_ind, module_k[layer_ind])] = outs[layer_ind]
        handle_list[layer_ind].remove()
    return feature_dict, output


def sample_act(neural_act: torch.tensor, layer_list: List, sample_size: int)-> Tuple[torch.tensor, List]:
    """
    Stratified sampling certain number of neurons' output given all activating vector of a model.
    Input args:
        neural_act (torch.tensor): n*d tensor. n is the total number of neurons and d is number of record (input sample size)
        layer_list (List): a list contain Conv2d or Linear module of a network. it is the return of parse_arch
        sample_size (int): Interger that specifies the number of neurons to be sampled
    """
    conv_nfilters_list=[x.in_channels for x in layer_list[0] if hasattr(x, "in_channels")]
    linear_nneurons_list=[x.in_features for x in layer_list[0] if hasattr(x, "in_features")]
    n_neurons_list=conv_nfilters_list+linear_nneurons_list
    layer_sample_num = [int(sample_size * x / sum(n_neurons_list)) for x in n_neurons_list]
    n_neurons_list=list(np.cumsum(n_neurons_list))
    # Stratified sampling for each layer
    n_neurons_list=[0]+n_neurons_list
    sample_ind=[np.random.choice(range(n_neurons_list[i], n_neurons_list[i+1]), layer_sample_num[i], replace=False) for i in range(len(n_neurons_list)-1) if layer_sample_num[i]]
    sample_n_neurons_list=[len(x) for x in sample_ind]
    sample_ind=np.concatenate(sample_ind)

    return neural_act[sample_ind], sample_n_neurons_list


def process_pd(pd: torch.tensor, layer_list: List, sample_n_neurons_list: List=None)-> torch.tensor:
    if not sample_n_neurons_list:
        # If the target sampling neurons list is not given then set it to be the whole layer_list
        conv_nfilters_list=[x.in_channels for x in layer_list[0] if hasattr(x, "in_channels")]
        linear_nneurons_list=[x.in_features for x in layer_list[0] if hasattr(x, "in_features")]
        n_neurons_list=conv_nfilters_list+linear_nneurons_list
        n_neurons_list=[0]+list(np.cumsum(n_neurons_list))
    else:
        n_neurons_list=[0]+list(np.cumsum(sample_n_neurons_list))
    maxpool_pd=np.zeros([len(n_neurons_list)-1, len(n_neurons_list)-1])
    for i in range(len(n_neurons_list)-1):
        for j in range(i, len(n_neurons_list)-1):
            if i==j:
                maxpool_pd[i,j]=1
            else:
                block=pd[n_neurons_list[i]:n_neurons_list[i+1], n_neurons_list[j]:n_neurons_list[j+1]]
                # maxpool_pd[i,j]=block.max()
                block=block.flatten()
                per_ind=np.argpartition(block.flatten(), -int(0.4*len(block)))[-int(0.4*len(block)):]
                maxpool_pd[i,j]=block[per_ind].mean()
                maxpool_pd[j,i]=maxpool_pd[i,j]
    return maxpool_pd


def discorr_adjacency_gpu(X: torch.tensor, Y: torch.tensor = None)-> torch.tensor:
    """
    Distance-correlation matrix calculation in tensor format.

    Dist-corr between two vector a and b is:

        dist-corr(v_1, v_2) = (1/n^2)\sum_{i=1}^n \sum_{j=1}^n A_{i,j} B_{i, j}

        where:
            A_{i,j} = a_{i,j} - a_{i, .} - a_{., j} + a_{., .}
            B_{i,j} = b_{i,j} - b_{i, .} - b_{., j} + b_{., .}

            a_{i,j} = |a_i - a_j|_p
            b_{i,j} = |b_i - b_j|_p
            a_{i, .} = (1/n) sum_{j=1}^n a_{i, j}
            a_{., j} = (1/n) sum_{i=1}^n a_{i, j}
            a_{., .} = (1/n^2) sum_{i=1}^n sum_{j=1}^n a_{i, j}

    Input args:
        X (torch.tensor). n*d. n is the number of neurons and d is the feature dimension.
        Y (torch.tensor). Optional.
    """
    n, m = X.shape
    # If Y is not given, then calculate distcorr(X, X)
    if not Y:
        Y = X
    # Constrain the size of tensor to be sent to GPU to avoid memory overflow
    # Con-comment to use GPU
    # if (64*n**2)/(10**9) < 8:
    #     X = X.cuda()
    #     Y = Y.cuda()
    bpd = torch.cdist(X.unsqueeze(2), Y.unsqueeze(2), p=2)
    bpd = bpd - bpd.mean(axis=1)[:, None, :] - bpd.mean(axis=2)[:, : , None] + bpd.mean((1, 2))[:, None, None]
    pd = torch.mm(bpd.view(n, -1), bpd.view(n, -1).T)
    del bpd, X, Y
    gc.collect()
    torch.cuda.empty_cache()

    pd/=n**2
    pd=torch.sqrt(pd)
    pd/=(torch.sqrt(torch.diagonal(pd)[None, :]*torch.diagonal(pd)[:, None])+1e-8)
    pd = pd.detach().cpu().numpy()
    np.fill_diagonal(pd, 1)

    return pd

def neural_corr(model: torch.nn.Module, train_loader: data.DataLoader)-> Tuple[torch.tensor, List]:
    """
    Build neural correlation matrix for the input model.
    Input args:
        model (torch.nn.Module): torch network
        train_loader (torch.utils.data.DataLoader): input images dataloader
    Return:
        pairwise_corr_mat (torch.tensor): pairwise correlation matrix
        sample_neurons_list (List): sampled neural activating vectors
    """

    # used to collect intermediate output for both clean input images and psf input images
    feature_dict_clean = defaultdict(list)
    feature_dict_troj = defaultdict(list)

    for _, (ind, images_c, images_t, _, _) in enumerate(train_loader):
        images_c, images_t = images_c.cuda(), images_t.cuda()
        feature_dict_c = feature_collect(model, images_c)
        feature_dict_t = feature_collect(model, images_t)
        for k in feature_dict_c:
            feature_dict_clean[k] += feature_dict_c[k]
            feature_dict_troj[k] += feature_dict_t[k]

    orig_neural_act = []
    troj_neural_act = []
    for k in feature_dict_clean:

        # Format PSF input's activating vectors
        if len(feature_dict_clean[k][0].shape) == 3:
            # If Conv2d layer then treat the maximum value as the activating value of the filter
            layer_act = [feature_dict_clean[k][i].max(1)[0].max(1)[0].unsqueeze(1) for i in range(len(feature_dict_clean[k]))]
        else:
            layer_act = [feature_dict_clean[k][i].unsqueeze(1) for i in range(len(feature_dict_clean[k]))]

        if len(feature_dict_clean[k][0].shape) == 3:
            layer_act = torch.cat(layer_act, dim=1)
            layer_act = layer_act.values
        else:
            layer_act = torch.cat(layer_act, dim=1)
        orig_neural_act.append(layer_act)

        if len(feature_dict_troj[k][0].shape) == 3:
            layer_act = [feature_dict_troj[k][i].max(1)[0].max(1)[0].unsqueeze(1) for i in range(len(feature_dict_troj[k]))]
        else:
            layer_act = [feature_dict_troj[k][i].unsqueeze(1) for i in range(len(feature_dict_troj[k]))]

        if len(feature_dict_troj[k][0].shape) == 3:
            layer_act = torch.cat(layer_act, dim=1)
            layer_act = layer_act.values
        else:
            layer_act = torch.cat(layer_act, dim=1)
        troj_neural_act.append(layer_act)

    merge_neural_act = torch.cat((orig_neural_act, troj_neural_act), dim=1)

    layer_list = parse_arch(model)
    conv_nfilters_list = [x.in_channels for x in layer_list[0] if hasattr(x, "in_channels")]
    linear_nneurons_list = [x.in_features for x in layer_list[0] if hasattr(x, "in_features")]
    sample_neurons_list = None

    if len(merge_neural_act) > SAMPLE_LIMIT:
        merge_neural_act, sample_neurons_list = sample_act(merge_neural_act, layer_list, sample_size=SAMPLE_LIMIT)
    pairwise_corr_mat = discorr_adjacency_gpu(merge_neural_act)

    return pairwise_corr_mat, sample_neurons_list
