import torch
import numpy as np
import pandas as pd
import cv2

from sklearn.metrics import roc_auc_score
import xgboost as xgb

import os
import sys
import argparse
import json
from collections import defaultdict
import pickle as pkl
from datetime import date
import glob
import re
import time
import random

# For my own experiment only
sys.path.append("/home/songzhu/PycharmProjects/Troj/baseline/Topo/dnn-topology/paper/trojai/trojai")
import modelgen
from networks import *
from topological_feature_extractor import *
from run_crossval import *

meta_path = "/data/trojanAI/round1/METADATA.csv"
metadata = pd.read_csv(meta_path)
model_name_list = metadata['model_name']

small_trigger_ind = model_name_list[(metadata['model_architecture'].str.contains('resnet')) & \
                                    (metadata['triggered_fraction']<=0.12) & \
                                    (metadata['triggered_fraction']>=0.08) | \
                                    (metadata['triggered_fraction']==0.00)].tolist()
large_trigger_ind = model_name_list[(metadata['model_architecture'].str.contains('resnet')) & \
                                    (metadata['triggered_fraction']<=0.55) & \
                                    (metadata['triggered_fraction']>=0.45) | \
                                    (metadata['triggered_fraction']==0.00)].tolist()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ind
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

    psf_config = {}
    psf_config['step_size'] = int(args.step_size)
    psf_config['stim_level'] = int(args.stim_level)
    psf_config['patch_size'] = int(args.patch_size)
    inputsize = [int(x) for x in args.input_size.split(',')]
    psf_config['input_shape'] = (inputsize[0], inputsize[1], inputsize[2])
    inputrange = [int(x) for x in args.input_range.split(',')]
    psf_config['input_range'] = (inputrange[0], inputrange[1])
    psf_config['device'] = device

    root = args.data_root
    meta=os.path.join(root, 'METADATA.csv')
    meta=pd.read_csv(meta)
    min_num_classes=meta['number_classes'].unique().max()
    model_list = sorted(glob.glob(os.path.join(root, "**/id-"+'[0-9]'*8)))
    # ---------------- Data Extraction Part ---------------------------------
    gt_list = []
    fv_list = []
    arch_list=[]
    model_name_list=[]
    # First extract feature for all provided models
    print('-------------- Feature Extraction Process for All Models ----------------')
    for j in range(len(model_list)):
    # for j in range(500, 750):
        model_path=model_list[j]
        model_name = model_path.split('/')[-1]

        if (args.trigger_size == 'small') and (not model_name in small_trigger_ind):
            continue
        if (args.trigger_size == 'larger') and (not model_name in large_trigger_ind):
            continue

        # arch_list.append(meta['model_architecture'][meta['model_name']==model_name].iloc[0])
        # if not ('resnet' in meta['model_architecture'][meta['model_name']==model_name].iloc[0]):
        #     continue
        # if not meta['number_classes'][meta['model_name']==model_name].iloc[0]==16:
        #     continue
        # if not meta['trigger_type'][meta['model_name']==model_name].iloc[0] in ['None', 'polygon']:
        #     continue

        # if (meta['triggers_0_type'][meta['model_name']==model_name].iloc[0] in ['instagram']) or \
        # (meta['triggers_1_type'][meta['model_name']==model_name].iloc[0] in ['instagram']):
        #     continue

        # if (meta['triggers_0_type'][meta['model_name']==model_name].iloc[0] in ['instagram']):
        #     continue

        t0=time.time()
        model_file_path = None
        model_train_example_folder = None
        for root_m, dirnames, filenames in os.walk(model_list[j]):
            for filename in filenames:
                if filename.endswith('.pt'):
                    model_file_path=os.path.join(root_m, filename)
                    break
            for dirname in dirnames:
                if dirname=='example_data':
                    model_train_example_folder = os.path.join(root_m, dirname)
                    break
            if model_file_path and model_train_example_folder:
                break

        try:
            model = torch.load(model_file_path).to(device)
        except:
            print("Model {} .pt file is missing, skip to next model".format(model_name))
            continue
        model.eval()

        if args.partial and (not (model._get_name() in [args.network])):
            continue

        # if not (model._get_name() in ['ResNet']):
        #     continue

        # if not (meta['model_architecture'][meta['model_name']==model_name].iloc[0] in ['resnet18', 'resnet34', 'resnet50']):
        #     continue

        save_file_path = os.path.join('/'.join(model_file_path.split('/')[:-1]), 'extracted_psf_topo_feature.pkl')
        print(save_file_path)
        if os.path.exists(save_file_path):
            fv_m=pkl.load(open(save_file_path, 'rb'))
            fv_list.append(fv_m)
            print("Extract Feature from Model-{} --- Use Time {:.3f}".format(model_name, time.time() - t0))
            if 'poisoned' in meta.columns:
                gt = meta['poisoned'][meta['model_name'] == model_name].iloc[0]
            else:
                gt = meta['ground_truth'][meta['model_name'] == model_name].iloc[0]
            gt_list.append(gt)
            model_name_list.append(model_name)
            continue
        else:
            continue

        use_examples = args.use_examples
        img_c = None
        pattern=re.compile('class_([0-9])*.*.png')
        if use_examples and os.path.exists(model_train_example_folder):
            img_c = defaultdict(list)
            example_files = os.listdir(model_train_example_folder)
            # Shuffle the example images
            rand_ind=np.random.choice(range(len(example_files)), len(example_files), replace=False)
            example_files=[example_files[i] for i in rand_ind]
            n_classes = meta['number_classes'][meta['model_name']==model_name].iloc[0]
            for i in range(len(example_files)):
                example=example_files[i]
                c=int(pattern.findall(example)[0])
                if not len(img_c[c]):
                    img_file=os.path.join(model_path, 'example_data', example)
                    img = torch.from_numpy(cv2.imread(img_file, cv2.IMREAD_UNCHANGED)).float()
                    img_c[c].append(img.unsqueeze(0))
                total_examples = sum([len(img_c[c]) for c in img_c])
                if len(img_c.keys()) == n_classes and total_examples == n_classes:
                    break

        fv = topo_psf_feature_extract_v1(model, img_c, psf_config)
        with open(save_file_path, 'wb') as f:
            pkl.dump(fv, f)
        f.close()
        fv_list.append(fv)

        print("Extract Feature from Model-{} --- Use Time {:.3f}".format(model_name, time.time()-t0))

    # return (None, None, None)
    # TODO: change feature processing method to accommodate later round
    # What is the best processing here ?
    # fv_list[i]['psf_feature_pos'] shape: 2 * nExample * fh * fw * nStimLevel * nClasses
    if args.classifier=='xgboost':
        # psf_fv=[]
        # for i in range(len(fv_list)):
        #     psf_fv_pos_i=fv_list[i]['psf_feature_pos']
        #     _, nEx, fh, fw, nSim, C=psf_fv_pos_i.shape
        #     psf_fv_pos_i=psf_fv_pos_i.reshape(2, nEx, -1, nSim, C)
        #     psf_diff=(psf_fv_pos_i.max(dim=3)[0]-psf_fv_pos_i.min(dim=3)[0]).max(2)[0]
        #     psf_med=psf_fv_pos_i.median(dim=3)[0].max(2)[0]
        #     psf_std=psf_fv_pos_i.std(dim=3).max(2)[0]
        #     psf_topk=psf_fv_pos_i.topk(k=min(3, nSim), dim=3)[0].mean(2).max(2)[0]
        #     psf_diff_mean_max=psf_diff.max(dim=1)[0].mean(dim=1)
        #     psf_diff_max_mean=psf_diff.mean(dim=1).max(dim=1)[0]
        #     psf_diff_max_max=psf_diff.max(dim=1)[0].max(dim=1)[0]
        #     psf_diff_mean_mean=psf_diff.mean(dim=(1,2))
        #     psf_diff_mean_std=psf_diff.mean(dim=1).std(dim=1)
        #     psf_diff_max_std=psf_diff.max(dim=1)[0].std(dim=1)
        #     psf_med_mean_max = psf_med.max(dim=1)[0].mean(dim=1)
        #     psf_med_max_mean = psf_med.mean(dim=1).max(dim=1)[0]
        #     psf_med_max_max = psf_med.max(dim=1)[0].max(dim=1)[0]
        #     psf_med_mean_mean = psf_med.mean(dim=(1,2))
        #     psf_med_mean_std = psf_med.mean(dim=1).std(dim=1)
        #     psf_med_max_std = psf_med.max(dim=1)[0].std(dim=1)
        #     psf_std_mean_max = psf_std.max(dim=1)[0].mean(dim=1)
        #     psf_std_max_mean = psf_std.mean(dim=1).max(dim=1)[0]
        #     psf_std_max_max = psf_std.max(dim=1)[0].max(dim=1)[0]
        #     psf_std_mean_mean = psf_std.mean(dim=(1,2))
        #     psf_std_mean_std = psf_std.mean(dim=1).std(dim=1)
        #     psf_std_max_std = psf_std.max(dim=1)[0].std(dim=1)
        #     psf_topk_mean_max = psf_topk.max(dim=1)[0].mean(dim=1)
        #     psf_topk_max_mean = psf_topk.mean(dim=1).max(dim=1)[0]
        #     psf_topk_max_max = psf_topk.max(dim=1)[0].max(dim=1)[0]
        #     psf_topk_mean_mean = psf_topk.mean(dim=(1,2))
        #     psf_topk_mean_std = psf_topk.mean(dim=1).std(dim=1)
        #     psf_topk_max_std = psf_topk.max(dim=1)[0].std(dim=1)
        #     psf_fv_pos_i=torch.cat([psf_diff_mean_max, psf_diff_max_mean, psf_diff_max_max, psf_diff_mean_mean, psf_diff_mean_std, psf_diff_max_std,
        #                                psf_med_mean_max, psf_med_max_mean, psf_med_max_max, psf_med_mean_mean, psf_med_mean_std, psf_med_max_std,
        #                                psf_std_mean_max, psf_std_max_mean, psf_std_max_max, psf_std_mean_mean, psf_std_mean_std, psf_std_max_std,
        #                                psf_topk_mean_max, psf_topk_max_mean, psf_topk_max_max, psf_topk_mean_mean, psf_topk_mean_std, psf_topk_max_std]).unsqueeze(0)
        #     psf_fv.append(psf_fv_pos_i)
        # psf_feature_dat=torch.cat(psf_fv)
        #
        # topo_feature = torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
        # topo_feature[np.where(topo_feature == np.Inf)] = 1
        # dat = torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
        # # dat=topo_feature.view(topo_feature.shape[0], -1)
        # # dat=topo_feature.max(dim=2)[0]
        # # dat=psf_feature_dat
        # dat=preprocessing.scale(dat)
        # gt_list = torch.tensor(gt_list)
        psf_feature=[]
        for i in range(len(fv_list)):
            _, nEx, fnW, fnH, nStim, C = fv_list[i]['psf_feature_pos'].shape
            select_c=torch.randperm(C)[:5]
            fv_list_i=fv_list[i]['psf_feature_pos'][:,:,:,:,:,select_c].unsqueeze(0)
            psf_feature.append(fv_list_i)
        psf_feature=torch.cat(psf_feature)
        # psf_feature=torch.cat([fv_list[i]['psf_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
        topo_feature=torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
        topo_feature[np.where(topo_feature==np.Inf)]=1
        n, _, nEx, fnW, fnH, nStim, _ = psf_feature.shape
        psf_feature_dat=psf_feature.reshape(n, 2, -1, nStim, 5)
        psf_diff_max=(psf_feature_dat.max(dim=3)[0]-psf_feature_dat.min(dim=3)[0]).max(2)[0].view(len(gt_list), -1)
        psf_med_max=psf_feature_dat.median(dim=3)[0].max(2)[0].view(len(gt_list), -1)
        psf_std_max=psf_feature_dat.std(dim=3).max(2)[0].view(len(gt_list), -1)
        psf_topk_max=psf_feature_dat.topk(k=2, dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
        psf_feature_dat=torch.cat([psf_diff_max, psf_med_max, psf_std_max, psf_topk_max], dim=1)

        dat=torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
        # dat=topo_feature.view(topo_feature.shape[0], -1)
        # dat=topo_feature.max(dim=2)[0].view(len(fv_list), -1)
        # dat=psf_feature_dat
        dat=preprocessing.scale(dat)
        gt_list=torch.from_numpy(np.array(gt_list))
        #
        N = len(fv_list)
        n_train = int(args.train_test_split * N)
        ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
        train_ind = ind_reshuffle[:n_train]
        test_ind = ind_reshuffle[n_train:]

        feature_train, feature_test = dat[train_ind], dat[test_ind]
        gt_train, gt_test = gt_list[train_ind], gt_list[test_ind]
        # Run the training and hyper-parameter searching process
        print('Running hyper-parameter searching and training')
        best_model_list = run_crossval_xgb(np.array(feature_train), np.array(gt_train))

        with open('./temp/resnet18_round1_{}_best_model_list_{}.pkl'.format('large', seed), 'wb') as f:
            pkl.dump(best_model_list, f)
        f.close()
        #
        with open('./temp/resnet18_round1_{}_best_model_list_{}.pkl'.format('large', seed), 'rb') as f:
            best_model_list = pkl.load(f)
        f.close()

        # Evaluation stage
        feature = feature_test
        labels = np.array(gt_test)
        dtest = xgb.DMatrix(np.array(feature), label=labels)
        y_pred = 0
        for i in range(len(best_model_list['models'])):
            best_bst=best_model_list['models'][i]
            weight=best_model_list['weight'][i]/sum(best_model_list['weight'])
            y_pred += best_bst.predict(dtest)*weight
        # y_pred = y_pred / len(best_model_list['models'])
        # Use sigmoid transformation
        # T, b = best_model_list['threshold']
        # y_pred=torch.sigmoid(b*torch.tensor(y_pred-T)).numpy()
        acc_test = np.sum((y_pred >= 0.5) == labels) / len(y_pred)
        auc_test = roc_auc_score(labels, y_pred)
        ce_test = np.sum(-(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))) / len(y_pred)

    if args.classifier=='mlp':
        dat=[]
        # TODO: not necessary needed
        _, nEx_0, fh_0, fw_0, nSim_0, C_0=fv_list[0]['psf_feature_pos'].shape
        for i in range(len(fv_list)):
            psf_fv_pos_i=fv_list[i]['psf_feature_pos']
            _, nEx, fh, fw, nSim, C=psf_fv_pos_i.shape

            if (not nEx==nEx_0) or (not fh==fh_0) or (not fw==fw_0) or (not nSim==nSim_0):
                continue

            psf_fv_pos_i=psf_fv_pos_i.permute(5, 0, 1, 2, 3, 4)
            psf_fv_pos_i=psf_fv_pos_i.reshape(C, -1)
            # psf_diff=(psf_fv_pos_i.max(dim=3)[0]-psf_fv_pos_i.min(dim=3)[0]).max(2)[0].permute(1,0,2)
            # psf_med=psf_fv_pos_i.median(dim=3)[0].max(2)[0].permute(1,0,2)
            # psf_std=psf_fv_pos_i.std(dim=3).max(2)[0].permute(1,0,2)
            # psf_topk=psf_fv_pos_i.topk(k=3, dim=3)[0].mean(2).max(2)[0].permute(1,0,2)
            # psf_diff_max=psf_diff.max(dim=2)[0]
            # psf_diff_min=psf_diff.min(dim=2)[0]
            # psf_diff_mean=psf_diff.mean(dim=2)
            # psf_diff_std=psf_diff.std(dim=2)
            # psf_med_max=psf_med.max(dim=2)[0]
            # psf_med_min=psf_med.min(dim=2)[0]
            # psf_med_mean=psf_med.mean(dim=2)
            # psf_med_std=psf_med.std(dim=2)
            # psf_std_max=psf_std.max(dim=2)[0]
            # psf_std_min=psf_std.min(dim=2)[0]
            # psf_std_mean=psf_std.mean(dim=2)
            # psf_std_std=psf_std.std(dim=2)
            # psf_topk_max=psf_topk.max(dim=2)[0]
            # psf_topk_min=psf_topk.min(dim=2)[0]
            # psf_topk_mean=psf_topk.mean(dim=2)
            # psf_topk_std=psf_topk.std(dim=2)
            topo_fv_pos_i=fv_list[i]['topo_feature_pos'].view(nEx, -1)
            # dat_pos_i = torch.cat([psf_diff_max, psf_diff_min, psf_diff_mean, psf_diff_std,
            #                           psf_med_max, psf_med_min, psf_med_mean, psf_med_std,
            #                           psf_std_max, psf_std_min, psf_std_mean, psf_std_std,
            #                           psf_topk_max, psf_topk_min, psf_topk_mean, psf_topk_std,
            #                           topo_fv_pos_i], dim=1)
            # dat_pos_i = topo_fv_pos_i
            dat_pos_i={'psf_fv_pos_i':psf_fv_pos_i, 'topo_fv_pos_i':topo_fv_pos_i}
            dat.append(dat_pos_i)

        N = len(dat)
        n_train = int(args.train_test_split * N)
        ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
        train_ind = ind_reshuffle[:n_train]
        test_ind = ind_reshuffle[n_train:]

        feature_train = [dat[i] for i in train_ind]
        feature_test = [dat[i] for i in test_ind]
        gt_train = [gt_list[i] for i in train_ind]
        gt_test = [gt_list[i] for i in test_ind]

        # Run the training and hyper-parameter searching process
        print('Running hyper-parameter searching and training')
        best_model_list = run_crossval_mlp(feature_train, gt_train)

        # Evaluation
        output_mv=torch.zeros(len(feature_test), 2).cuda()
        for i in range(len(best_model_list['models'])):
            psf_encoder, topo_encoder, cls = best_model_list['models'][i]
            weight = best_model_list['weight'][i]/sum(best_model_list['weight'])

            psf_encoder.eval()
            topo_encoder.eval()
            cls.eval()
            correct = 0
            total = 0
            for j in range(0, max(int((len(feature_test) - 1) / 32), 1)):
                batch = feature_test[(32*j):min(32*(j+1), len(feature_test))]
                embedding_list = []
                for single_input in batch:
                    psf_fv_pos_i=single_input['psf_fv_pos_i'].cuda()
                    topo_fv_pos_i=single_input['topo_fv_pos_i'].cuda()
                    psf_embedding=psf_encoder(psf_fv_pos_i)
                    if len(topo_fv_pos_i)==1:
                        topo_fv_pos_i=topo_fv_pos_i.repeat(5, 1)+torch.randn(5, topo_fv_pos_i.shape[1]).cuda()
                    topo_embedding=topo_encoder(topo_fv_pos_i)
                    embedding=torch.cat([psf_embedding.mean(0).flatten(), topo_embedding.mean(0).flatten()])
                    embedding_list.append(embedding)
                embeddings = torch.cat([x.unsqueeze(0) for x in embedding_list])
                output = cls(embeddings)
                output_mv[(32*j):min(32*(j+1), len(feature_test))]+=output*weight

        gt_test=torch.tensor(gt_test)
        output=output_mv
        score=torch.nn.functional.softmax(output, 1).detach().cpu()
        pred = score.argmax(1)
        correct += pred.eq(gt_test).sum().item()
        total += len(gt_test)
        acc_test = correct / total
        auc_test = roc_auc_score(gt_test.detach().cpu().numpy(), score[:, 1].numpy())
        ce_test = -np.mean(np.array(gt_test)*np.log(np.maximum(score[:,1].numpy(), 1e-4))+(1-np.array(gt_test))*np.log(np.maximum(1-score[:,1].numpy(), 1e-4)))

    # TODO: Change logger naming method
    logger_name=date.today().strftime("%d-%m-%Y")+'_'+args.dataset+'_'+"-".join([str(x) for x in psf_config['input_shape']])
    logger_file=os.path.join(args.log_path, logger_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    logger=open(logger_file, 'w')
    print("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test*100, auc_test, ce_test))
    logger.write("Final Acc {:.3f}% - Final AUC {:.3f} - Fianl CE {:.3f}".format(acc_test*100, auc_test, ce_test))
    logger.flush()
    logger.close()

    return acc_test, auc_test, ce_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract feature and train trojan detector for NIST dataset experiment')
    parser.add_argument('--data_root', type=str, help='Root folder that saves the experiment models')
    parser.add_argument('--step_size', type=float, help='Stimulation stepsize used in PSF')
    parser.add_argument('--stim_level', type=int, help='Number of stimulation levels')
    parser.add_argument('--patch_size', type=int, help='Stimulation patch size')
    parser.add_argument('--input_size', type=str, help='Training dataset used in this experiment', default='1,28,28')
    parser.add_argument('--input_range', type=str, help='Input data value range', default='0, 255')
    parser.add_argument('--use_examples', type=bool, help='Whether use example input or not', default=False)
    parser.add_argument('--classifier', type=str, help='trojan classifier type', default='xgboost', choices=('xgboost', 'mlp'))
    parser.add_argument('--train_test_split', type=float, help='Train and test split ratio', default=0.2)
    parser.add_argument('--n_folds', type=int, help='Cross validation folds', default=5)
    parser.add_argument('--hpsearch_budget', type=int, help='Hyper-parameter search budget', default=20)
    parser.add_argument('--log_path', type=str, help='Output log save dir', default='./tmp')
    parser.add_argument('--gpu_ind', type=str, help='Indices of GPUs to be used', default='6')
    parser.add_argument('--partial', type=bool, help='Whether use partial data', default=True)
    args = parser.parse_args()

    # For my own experiment purpose
    args.data_root = '/data/trojanAI/round1'
    args.dataset='round1'
    args.network='ResNet'
    args.trigger_size = 'small'
    args.step_size = 13
    args.stim_level = 32
    args.patch_size = 13
    args.gpu_ind = '2'
    args.input_size = '3,224,224'
    args.input_range = '0, 1020'
    args.use_examples = False
    args.train_test_split=0.8
    args.log_path='/home/songzhu/PycharmProjects/Troj/baseline/Topo/dnn-topology/paper/experiment_log'

    repeat_exp_logfile=date.today().strftime("%d-%m-%Y")+'_'+'Topo'+'_'+args.dataset+'_'+args.network+'.json'
    repeat_exp_logfile=os.path.join(args.log_path, repeat_exp_logfile)
    exp_config={}
    for k, v in args._get_kwargs():
        exp_config[k]=v

    repeat_time=3
    acc_list=[]
    auc_list=[]
    ce_list=[]
    for i in range(repeat_time):
        acc_test, auc_test, ce_test = main(args)
        acc_list.append(acc_test)
        auc_list.append(auc_test)
        ce_list.append(ce_test)

        exp_config['acc_list']=acc_list
        exp_config['auc_list']=auc_list
        exp_config['ce_list']=ce_list

        with open(repeat_exp_logfile, 'w') as f:
            json.dump(exp_config, f, sort_keys=False, indent=4)
        f.close()
