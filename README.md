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
If you want to use your own Trojaned networks database, make sure your folder is in following structure: 

    .
    ├── id-00000000                 # Model ID
        ├── model.pt.1              # Model .pt file
        ├── configure.csv           # Model meta file
        ├── gt.txt                  # Trojaned Flag
        └── examples                # Folder that contains clean input images (optional)
            ├── example_1.png
            ├── example_2.png
            ├── example_3.png
            ├── ...
    ├── id-00000001
    ├── id-00000002
    ├── ...

## Running TopoTrojanDetection
Under construction