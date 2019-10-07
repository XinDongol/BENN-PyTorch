# BENN on ImageNet dataset

### Prerequisites

To set up environment for this code, you will need,

* [Pytorch](https://pytorch.org/)==0.3.1/0.4.0
* [Python Cafffe](https://caffe.berkeleyvision.org/installation.html)

The easiest way to set up environment is to use `Docker`. We suggest use images from [`ufoym/deep`](https://github.com/ufoym/deepo).  The image with tag `all-py27-cu90` is a good choice to reproduce our results. 

### Dataset Preparation

* Before training and/or testing, please download the `LMDB` dataset from [here](https://drive.google.com/uc?export=download&id=0B-7I62GOSnZ8aENhOEtESVFHa2M) and unzip it. (It will need ~200G disk space.)
* We are using `Python Caffe` API to load the preprocessed dataset. If you use different data source with different preprocessing, validation accuracy may not come up to expectation with our provided checkpoints. However, you can train your own models with our method for sure. 

Open docker env

```bash
docker run --runtime=nvidia -it --ipc=host -v ~/:/data ufoym/deepo:all-py27-cu90 bash
```

Check the version of Pytorch

```bash
pip install torch==0.4.0 --upgrade
```

Download [initial pretrained networks](https://drive.google.com/file/d/1m1aIPhWAz1-yjLejZAfIqeO7YW1QCwbE/view?usp=sharing) and put it into `./networks`.


### Train BENN on ImageNet

`cd scripts` and then run:

Bagging:
```bash
python2 main_bagging_imagenet.py [-arguments]
```

Boosting:
```bash
python2 main_boosting_imagenet.py [-arguments]
```

You may also want to modify the hyperparameters in the arguments such as learning 
rate or number of epochs to test different training setups.


