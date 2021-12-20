# TopoTrojanDetection

This is the code for the paper:

**<a> Topological Detection of Trojaned Neural Networks </a>**
<br>
Songzhu Zheng, Yikai Zhang, Hubert Wagner, Mayank Goswami, Chao Chen
</br>
[[Paper Link]](https://openreview.net/pdf?id=1r2EannVuIA).
Presented at [NeurIPS 2021](https://nips.cc/virtual/2021/poster/26328)

If you find this code useful in your research please cite:
```
@inproceedings{songzhu2021_Topo,
  title={Topological Detection of Trojaned Neural Networks},
  author={Songzhu Zheng, Yikai Zhang, Hubert Wagner, Mayank Goswami,  Chao Chen},
  booktitle={NeurIPS},
  year={2021}
}
```

## Introduction 

TopoTrojanDetection is a tool to build a classifier for Trojaned network detection. 
For a given pool of networks that are under scrutinization, 
this algorithm first extracts 6 topological features from each dimension 
(total number of points in persistent diagram, average persistence, average middle life, maximum middle life, maximum persistence, topk persistence) 
from each of these models. Together with these topological features, we also extract pixel-wise stimulation feature (logits and confidence of perturbed images). 
Next, a binary classifier is trained using these features to distinguish Trojaned and clean networks. 

![pipeline_demo](https://github.com/pingqingsheng/TopoTrojDetection/blob/main/images/pipeline_demo.png)

## Environment Setting

The environment setup for TopoTrojanDetection is listed in environment.yml. To install, run:

```bash
conda env create -f environment.yml
source activate TopoDetect
```

## Generate Trojaned Networks (Data Preparation)
__(Use Our Database)__ We use [trojai toolkit](https://trojai.readthedocs.io/en/latest/) to generate Trojaned networks. 
We provide our experiment data through following Google doc link: [MNIST+LeNet5](), [MNIST+ResNet18](), [CIFAR10+ResNet18](), and [CIFAR10+DenseNet121](). 

__(Train New Trojaned Network)__ We also provide source code to generate Trojaned network. To generate Trojaned networks on MNIST, run:
```bash
cd data
python batch_model_generation_MNSIT.py 
[--gpu] 
[--console]
[--parallel]
[--network MODEL_ARCHITECTURE] 
[--num_models NUMBER_OF_MODELS] 
[--troj_frac FRACTION_OF_TROJANED_IMAGES]
[--target_class TARGET_CLASS]
[--gpu_ind GPU_INDICE]
```

To generate Trojaned networks on CIFAR10, use:
```bash
cd data
python batch_model_generation_CIFAR10.py
[--gpu]
[--console]
[--parallel]
[--early_stopping]
[--network MODEL_ARCHITECTURE] 
[--num_models NUMBER_OF_MODELS] 
[--troj_frac FRACTION_OF_TROJANED_IMAGES]
[--target_class TARGET_CLASS]
[--train_val_split TRAIN_VALID_SPLIT]
[--gpu_ind GPU_INDICE]
```

For example, to generate 20 Trojaned LeNet5 networks with 20% all-to-one attack with target class 0, run:
```bash
cd data
python batch_model_generation_MNIST.py --gpu --network leenet5 --num_models 20 --troj_frac 0.2 --target_class 0
```


__(Use Customized Database)__


## Running AdaCorr 

There are several command-line flags that you can use to configure the input/output and hyperparameters to be applied by AdaCorr.

```bash
python LRTcorrect.py 
[--dataset] 
[--network]  
[--noise_type]
[--noise_level]
[--lr] 
[--n_epochs] 
[--epoch_start] 
[--epoch_update] 
[--epoch_interval] 
[--every_n_epoch] 
[--two_stage] 
[--gpu] 
[--n_gpus] 
[--seed]
```

- `--dataset : The dataset to be used in the experiment. Current version supports mnist/cifar10/cifar100/pc. The default is mnist.`
- `--network : The network architecture to be used as the backbone. Available options are cnn/resnet18/resnet34/preact_resnet18/preact_resnet34/preact_resnet101/pc. The default is preact_resnet34.`
- `--noise_type : Choose noise type to be applied to data's labels. Options are uniform/asymmetric.`
- `--noise_level : Set the noise level. Must be in the range [0, 1].`
- `--lr : Learning Rate. The default is 1e-3.`
- `--n_epochs : Number of epochs to train.`
- `--epoch_start : The epoch that retroactive loss is introduced.`
- `--epoch_update : The epoch that start to perform correction.`
- `--epoch_interval : Interval between the update of retroactive loss.`
- `--every_n_epoch : Sliding window for calculating network confidence.`
- `--two_stage : Use two stage training. Retrain the network from scratch using corrected labels. Default is 0. Two enable this feature set it to be 1.`
- `--gpu : The index of GPU to be used. The default is 0. If torch.cuda.is_available() is false the CPU will be used.`
- `--n_gpus : Number of GPUS to be used. The default is 1.`
- `--seed : Set the random seeds for your experiment.`

For example, to run AdaCorr on CIFAR10 with 20% uniform noisy level with <img src="https://render.githubusercontent.com/render/math?math=L_{ce}"> introduced at epoch 25 and correction starting at 30, you could run:
```bash
python LRTcorrect.py --dataset cifar10 --network preact_resnet34 --noise_type uniform --noise_level 0.2 --n_epochs 180 --epoch_start 25 --epoch_update 30
```

Current version doesn't support user-customized dataset. If you wish to use AdaCorr on your own dataset, your need to write your own data loader and replace the one used in __LRTcorrect.py__. If you have any question, please contact <imzszhahahaha@gmail.com> for the issue of implementation.

## Parameter Setting for Experiment Results ##
The hyper-parameter setting used to reproduce the result in the paper is presented in Experiment_Log folder. You can simply run these bash file to view the results.
All commands is in following format:
```bash
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --noise_type uniform --noise_level 0.2 --lr 1e-3 --epoch_start 10 --epoch_update 15 --n_epochs 180 --n_gpus 1 --gpu 0
```
#### MNIST - Uniform Noise ####
```bash
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --noise_type uniform --noise_level 0.2 --lr 1e-3 --epoch_start 10 --epoch_update 15 --n_epochs 180 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --noise_type uniform --noise_level 0.4 --lr 1e-3 --epoch_start 10 --epoch_update 15 --n_epochs 180 --n_gpus 1 --gpu 0 
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --noise_type uniform --noise_level 0.6 --lr 1e-3 --epoch_start 10 --epoch_update 15 --n_epochs 180 --n_gpus 1 --gpu 0 
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --noise_type uniform --noise_level 0.8 --lr 1e-3  --epoch_start 5 --epoch_update 10 --n_epochs 180 --n_gpus 1 --gpu 0 
```
#### CIFAR10 - Uniform Noise ####
```bash
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --noise_type uniform --noise_level 0.2 --lr 1e-3 --n_epochs 180 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --noise_type uniform --noise_level 0.4 --lr 1e-3 --n_epochs 180 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --noise_type uniform --noise_level 0.6 --lr 1e-3 --n_epochs 180 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --noise_type uniform --noise_level 0.8 --lr 1e-3 --n_epochs 180 --epoch_start 20 --epoch_update 25 --gpu 0 --n_gpus 1
```
#### CIFAR100 - Uniform Noise ####
```bash
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type uniform --noise_level 0.2 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type uniform --noise_level 0.4 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type uniform --noise_level 0.6 --epoch_start 25 --epoch_update 30 --gpu 0 --n_gpus 1
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type uniform --noise_level 0.8 --epoch_start 30 --epoch_update 35 --gpu 0 --n_gpus 1
```
#### Point Cloud - Uniform Noise ####
```bash
python -W ignore LRTcorrect.py --dataset pc --network pc --n_epochs 180 --lr 2e-3 --noise_type uniform --noise_level 0.2 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset pc --network pc --n_epochs 180 --lr 2e-3 --noise_type uniform --noise_level 0.4 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset pc --network pc --n_epochs 180 --lr 2e-3 --noise_type uniform --noise_level 0.6 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset pc --network pc --n_epochs 180 --lr 2e-3 --noise_type uniform --noise_level 0.8 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
```

#### MNIST - Asymmetric Noise ####
```bash
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.2 --epoch_start 5 --epoch_update 10 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.3 --epoch_start 5 --epoch_update 10 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset mnist --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.4 --epoch_start 5 --epoch_update 10 --n_gpus 1 --gpu 0

```
#### CIFAR10 - Asymmetric Noise ####
```bash
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.2 --epoch_start 20 --epoch_update 25 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.3 --epoch_start 20 --epoch_update 25 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset cifar10 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.4 --epoch_start 20 --epoch_update 25 --n_gpus 1 --gpu 0
```
#### CIFAR100 - Asymmetric Noise ####
```bash
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.2 --epoch_start 25 --epoch_update 30 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.3 --epoch_start 25 --epoch_update 30 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset cifar100 --network preact_resnet34 --n_epochs 180 --lr 1e-3 --noise_type asymmetric --noise_level 0.4 --epoch_start 25 --epoch_update 30 --n_gpus 1 --gpu 0
```
#### Point Cloud - Asymmetric Noise ####
```bash
python -W ignore LRTcorrect.py --dataset pc --network pc --noise_type asymmetric --lr 2e-3 --noise_level 0.2 --n_epochs 180 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset pc --network pc --noise_type asymmetric --lr 2.5e-3 --noise_level 0.3 --n_epochs 180 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
python -W ignore LRTcorrect.py --dataset pc --network pc --noise_type asymmetric --lr 2.5e-3 --noise_level 0.4 --n_epochs 180 --epoch_start 10 --epoch_update 15 --n_gpus 1 --gpu 0
```

## Performance
![Experiment_Table_1](https://github.com/pingqingsheng/LRT/blob/master/images/exp_table1.png)
![Experiment_Table_2](https://github.com/pingqingsheng/LRT/blob/master/images/exp_table2.png)

## Algorithm
![LRT Algorithm](https://github.com/pingqingsheng/LRT/blob/master/images/alg_1.png)
![Training Algorithm](https://github.com/pingqingsheng/LRT/blob/master/images/alg_2.png)

## Reference
