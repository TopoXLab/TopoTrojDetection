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

![pipeline_demo](https://github.com/TopoXLab/TopoTrojDetection/blob/main/images/github_demo1.png)

## Environment Setting

The environment setup for TopoTrojanDetection is listed in environment.yml. To install, run:

```bash
conda env create -f environment.yml
source activate TopoDetect
```

## Generate Trojaned Networks (Data Preparation)
__(Use Our Database)__ We use [trojai toolkit](https://trojai.readthedocs.io/en/latest/) to generate Trojaned networks. 
We provide our experiment data through following Google doc link: [MNIST+LeNet5](https://drive.google.com/file/d/17KSmLv42X5LLUEGiJCn5KzHSq0fXcywv/view?usp=sharing), 
[MNIST+ResNet18](https://drive.google.com/file/d/1AnFbSAMU5c_DOWquy8mGweqMDZ3AVgr-/view?usp=sharing), 
[CIFAR10+ResNet18](https://drive.google.com/file/d/1RWCf0IN5_vle-lI4sFWV1WcLTDvcDECj/view?usp=sharing), and 
[CIFAR10+DenseNet121](https://drive.google.com/file/d/1meb_yQfhZmn2RGBEOGKgfwayPLx0wroC/view?usp=sharing). 

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

For example, to generate 20 Trojaned LeNet5 networks with 20% one-to-one attack with target class 0 (the source class will be target-1, if target is 0 then the source is NUM_CLASS-1), 
run:
```bash
cd data
python batch_model_generation_MNIST.py --gpu --network leenet5 --num_models 20 --troj_frac 0.2 --target_class 0
```

__(Download NIST TrojAI Competition Dataset).__ To download TrojAI competition dataset, use this [download page link](https://pages.nist.gov/trojai/docs/data.html)

__(Use Customized Database)__
If you want to use your own Trojaned networks database, make sure your folder is in following structure: 

    .
    ├── id-00000000                 # Model ID
        ├── model.pt.1              # Model .pt file
        ├── experiment_train.csv    # Examples' meta file (please follow the naming here)
        ├── gt.txt                  # Trojaned Flag       (please follow the naming here)
        └── examples                # Folder that contains clean input images (optional, needed if use_examples)
            ├── example_1.png       # Name of images can be arbitrary
            ├── example_2.png
            ├── example_3.png
            ├── ...
    ├── id-00000001
    ├── id-00000002
    ├── ...

The ```configure.csv``` file should contain the absolute path to each of your training examples and 
flags indicating whether it's a Trojaned example or not. A example snippet of ```configure.csv``` is
shown below:

| path |  true_label | train_label | triggered |
|:---:|:---:|:---:|:---:|
|mnist_clean/train/mnist_train_id_0_class_5.png| 5 | 5 | False |
|mnist_clean/train/mnist_train_id_1_class_0.png| 0 | 0 | False |
|mnist_clean/train/mnist_train_id_2_class_4.png| 4 | 0 | True |
|...|||

## Running TopoTrojanDetection

There are also several hyper-parameter you can change in ```run_troj_detector.py``` to tune the algorithm. 

    STEP_SIZE:  int = 2 # Stimulation stepsize used in PSF
    PATCH_SIZE: int = 8 # Stimulation patch size used in PSF
    STIM_LEVEL: int = 4 # Number of stimulation level used in PSF
    INPUT_SIZE: List = [1, 28, 28] # Input images' shape (default to be MNIST)
    INPUT_RANGE: List = [0, 255]   # Input image range
    USE_EXAMPLE: bool =  False     # Whether clean inputs will be given or not
    TRAIN_TEST_SPLIT: float = 0.8  # Ratio of train to test
    
After you prepare your database, you are ready to run the Trojan detection training. Run following code to start the training: 

```bash
python run_troj_detector.py 
--data_root <DATABASE_PATH> 
[--log_path LOG_PATH] 
[--gpu_ind GPU_INDEICE]
[--seed RANDOM_SEED]
``` 
For example, after you run ```batch_model_generation_MNSIT.py```, you could run following line to start detector training: 
```bash
python run_troj_detector.py --data_root ./data/data
```


