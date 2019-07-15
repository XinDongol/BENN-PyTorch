# BENN on CIFAR-10 dataset

## Environment installation

Generally, we require:

  - python 2.7

  - numpy 1.16.4

  - opencv-python 4.1.0.25

  - pytorch 0.3.1

  - torchvision 0.2.0

Use the below command to install the environment:

`pip install -r requirements.txt`

## Retrain BENN models

The scripts of training AB and SB models of BENN are located in `./AB` and `./SB`. For example to retrain BENN-bagging (SB model):

`$ python main_bagging_SB.py --epochs 0 --retrain_epochs 100 --root_dir PATH/TO/YOUR/models_bagging_SB/`

To get advance control of the training process, refer to the argument parser in each script.

## Test pre-trained BENN models

First download the models from the [links](https://github.com/XinDongol/BENN-PyTorch), then run the corresponding python script to test pre-trained models and you should get the exact same numbers comparing with our [logs](https://github.com/XinDongol/BENN-PyTorch). For example to test BENN-bagging (SB model):

```$ python main_bagging_SB.py --epochs 0 --retrain_epochs 0 --root_dir PATH/TO/YOUR/DOWNLOADED/models_bagging_SB/```

**Notice:** For AB models, you should get around 79-82% accuracy for 32 ensembles. For SB models, you should get around 87-89% accuracy for 32 ensembles (usually 15-20 is a reasonable choice due to overfitting). The single BNN should have around 69-73% and 83-84% accuracy for AB and SB model respectively.
