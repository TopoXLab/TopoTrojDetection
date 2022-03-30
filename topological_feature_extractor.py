#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import copy
from collections import defaultdict
from typing import List, Dict

import torch
import numpy as np
from ripser import Rips
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import time

from topo_utils import mat_bc_adjacency, parse_arch, feature_collect, sample_act, mat_discorr_adjacency, mat_cos_adjacency, mat_jsdiv_adjacency, mat_pearson_adjacency


def makeSparseDM(D: np.array, threshold: float)-> np.array:
    """
    Convert a dense matrix to COO format. All values that are below thresh are set to be 0.
    Input args:
        D (np.array): matrix to be converted
        threshold (float): threshold below which value will be set to 0
    Return:
        matrix in compressed sparse column format
    """
    N = D.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= threshold]
    J = J[D <= threshold]
    V = D[D <= threshold]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()


def getGreedyPerm(D: np.array)-> List:
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Input args:
        D (np.array):  An NxN distance matrix for points
    Return:
        lamdas (List): list Insertion radii of all points
    """

    N = D.shape[0]
    # By default, takes the first point in the permutation to be the
    # first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]


def getApproxSparseDM(lambdas: List, eps: float, D: np.array)-> csr_matrix:
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight.
    Input args:
        lambdas (List): insertion radii for points
        eps (float): epsilon approximation constant
        D (np.array): NxN distance matrix, okay to modify because last time it's used
    Return:
        DSparse (scipy.sparse): A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()


def calc_topo_feature(PH: List, dim: int)-> Dict:
    """
    Compute topological feature from the persistent diagram.
    Input args:
        PH (List) : Persistent diagram
        dim (int) : dimension to be focused on
    Return:
        Dictionary contains topological feature
    """
    pd_dim = PH[dim]
    if dim == 0:
        pd_dim = pd_dim[:-1]
    pd_dim = np.array(pd_dim)
    betti = len(pd_dim)
    ave_persis = sum(pd_dim[:, 1] - pd_dim[:, 0]) / betti if betti > 0 else 0
    ave_midlife = (sum((pd_dim[:, 0] + pd_dim[:, 1]) / 2) / betti) if betti > 0 else 0
    med_midlife = np.median((pd_dim[:, 0] + pd_dim[:, 1]) / 2) if betti > 0 else 0
    max_persis = (pd_dim[:, 1] - pd_dim[:, 0]).max() if betti > 0 else 0
    top_5_persis = np.mean(np.sort(pd_dim[:, 1] - pd_dim[:, 0])[-5:]) if betti > 0 else 0
    topo_feature_dict = {"betti_" + str(dim): betti,
                         "avepersis_" + str(dim): ave_persis,
                         "avemidlife_" + str(dim): ave_midlife,
                         "maxmidlife_" + str(dim): med_midlife,
                         "maxpersis_" + str(dim): max_persis,
                         "toppersis_" + str(dim): top_5_persis}
    return topo_feature_dict


def topo_psf_feature_extract(model: torch.nn.Module, example_dict: Dict, psf_config: Dict)-> Dict:
    """
    Extract topological features from a given torch model.
    Input args:
        model (torch.nn.Module). Target model.
        example_dict (Dict). Optional. Dictionary contains clean input examples. If None then all blank images are used.
    Return:
        fv (Dict). Dictionary contains extracted features
    """
    step_size=psf_config['step_size']
    stim_level=psf_config['stim_level']
    patch_size=psf_config['patch_size']
    input_shape=psf_config['input_shape']
    input_valuerange=psf_config['input_range']
    n_neuron_sample=psf_config['n_neuron']
    method=psf_config['corr_method']
    device=psf_config['device']

    # If true input examples are not given, use all blank images instead
    if not example_dict:
        example_dict=defaultdict(list)
        example_dict[0].append(torch.zeros(input_shape).unsqueeze(0))

    model=model.to(device)
    test_input=example_dict[0][0].to(device)
    num_classes=int(model(test_input).shape[1])

    stim_seq=np.linspace(input_valuerange[0], input_valuerange[1], stim_level)
    # 2 represent score and conf
    feature_map_h=len(range(0, input_shape[1]-patch_size+1, step_size))
    feature_map_w=len(range(0, input_shape[2]-patch_size+1, step_size))
    # PSF feature dim : 2*m*h*w*L*C
    #  2: logits and confidence
    #  m: numebr of input examples
    #  h: feature map height
    #  w: feature map width
    #  L: number of stimulation levels
    #  C: number of classes
    psf_feature_pos=torch.zeros(
        2,
        len(example_dict.keys()),
        feature_map_h, feature_map_w,
        len(stim_seq), num_classes)
    # 12 is the number of topological features (including dim1 and dim2 features)
    topo_feature_pos=torch.zeros(
        len(example_dict.keys()),
        len(range(0, int(feature_map_h*feature_map_w))),
        12
    )

    PH_list=[]
    PD_list=[]
    rips = Rips(verbose=False)
    model=model.to(device)
    progress=0
    # For each class input examples, scan through pixels with step_size and modify corresponding pixel with different stimulation level.
    # Forward all these modified images to the network and collect output logits and confidence
    for c in example_dict:
        input_eg=copy.deepcopy(example_dict[c][0])
        feature_w_pos=0
        for pos_w in range(0, input_shape[1]-patch_size+1, step_size):
            feature_h_pos = 0
            for pos_h in range(0, input_shape[2]-patch_size+1, step_size):
                t0=time.time()
                count=0
                prob_input=input_eg.repeat(len(stim_seq),1,1,1)
                for i in stim_seq:
                    prob_input[count,:,
                               int(pos_w):min(int(pos_w+patch_size), input_shape[1]),
                               int(pos_h):min(int(pos_h+patch_size), input_shape[1])]=i
                    count+=1
                pred=[]
                batch_size=8 if len(prob_input)>=32 else 1
                if batch_size==1:
                    prob_input=prob_input.to(device)
                    feature_dict_c, output = feature_collect(model, prob_input)
                    pred.append(output.detach().cpu())
                else:
                    for b in range(int(len(prob_input)/batch_size)):
                        prob_input_batch=prob_input[(8*b):min(8*(b+1), len(prob_input))].to(device)
                        feature_dict_c, output = feature_collect(model, prob_input_batch)
                        pred.append(output.detach().cpu())
                pred=torch.cat(pred)
                psf_score=pred
                psf_conf=torch.nn.functional.softmax(psf_score, 1)

                psf_feature_pos[0, c, feature_w_pos, feature_h_pos]=psf_score
                psf_feature_pos[1, c, feature_w_pos, feature_h_pos]=psf_conf

                # Extract intermediate activating vectors
                neural_act = []
                for k in feature_dict_c:
                    if len(feature_dict_c[k][0].shape)==3:
                        layer_act = [feature_dict_c[k][i].max(1)[0].max(1)[0].unsqueeze(1) for i in range(len(feature_dict_c[k]))]
                    else:
                        layer_act = [feature_dict_c[k][i].unsqueeze(1) for i in range(len(feature_dict_c[k]))]
                    layer_act=torch.cat(layer_act, dim=1)
                    # Standardize the activation layer-wisely
                    layer_act=(layer_act-layer_act.mean(1, keepdim=True))/(layer_act.std(1, keepdim=True)+1e-30)
                    neural_act.append(layer_act)
                neural_act=torch.cat(neural_act)
                layer_list=parse_arch(model)
                sample_n_neurons_list=None
                if len(neural_act)>1.5e3:
                    neural_act, sample_n_neurons_list=sample_act(neural_act, layer_list, sample_size=n_neuron_sample)

                # Build neural correlation matrix
                if method=='distcorr':
                    neural_pd=mat_discorr_adjacency(neural_act)
                elif method=='bc':
                    neural_act=torch.softmax(neural_act, 1)
                    neural_pd=mat_bc_adjacency(neural_act)
                elif method=='cos':
                    neural_pd=mat_cos_adjacency(neural_act)
                elif method=='pearson':
                    neural_pd=mat_pearson_adjacency(neural_act)
                elif method=='js':
                    neural_act=torch.softmax(neural_act, 1)
                    neural_pd=mat_jsdiv_adjacency(neural_act)
                else:
                    raise Exception(f"Correlation metrics {method} doesn't implemented !")
                D=1-neural_pd.detach().cpu().numpy() if method!='bc' else -np.log(neural_pd.detach().cpu().numpy()+1e-6)
                PD_list.append(neural_pd.detach().cpu().numpy())

                # Approaximate sparse filtration to further save some computation
                if model._get_name=='ModdedLeNet5Net':
                    PH=rips.fit_transform(D, distance_matrix=True)
                else:
                    lambdas=getGreedyPerm(D)
                    D = getApproxSparseDM(lambdas, 0.1, D)
                    PH=rips.fit_transform(D, distance_matrix=True)

                PH[0]=np.array(PH[0])
                PH[1]=np.array(PH[1])
                PH[0][np.where(PH[0]==np.inf)]=1
                PH[1][np.where(PH[1]==np.inf)]=1
                PH_list.append(PH)
                # Compute the topological feature with the persistent diagram
                clean_feature_0=calc_topo_feature(PH, 0)
                clean_feature_1=calc_topo_feature(PH, 1)
                topo_feature=[]
                for k in sorted(list(clean_feature_0)):
                    topo_feature.append(clean_feature_0[k])
                for k in sorted(list(clean_feature_1)):
                    topo_feature.append(clean_feature_1[k])
                topo_feature=torch.tensor(topo_feature)
                topo_feature_pos[c, int(feature_w_pos*feature_map_w+feature_h_pos), :]=topo_feature
                feature_h_pos+=1

            feature_w_pos+=1

    fv={}
    fv['psf_feature_pos']=psf_feature_pos
    fv['topo_feature_pos']=topo_feature_pos
    fv['correlation_matrix']=np.vstack([x[None, :, :] for x in PD_list]).mean(0)
    return fv
