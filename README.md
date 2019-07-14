# BENN
Codes for Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?

CVPR 2019 [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.pdf)

If using the code, please cite our paper: [BibTex](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.html)

**Notice:** As mentioned in the paper (Section 7) we are aware of the overfitting problem caused by the ensemble technique. If retraining the models, they should basically match the results shown in the paper, but could be slightly higher or lower due to random initialization and overfitting.

## Train BENN on CIFAR-10 dataset

A customized Network-In-Network (NIN) model is used. Please see paper for architecture details.

| Ensemble   | Model | Train |     LR | BNN (start) | BENN (end) | Overfitting from | Best Voting |               Models Directory              |              Logs |
|------------|:-----:|:-----:|-------:|------------:|-----------:|:----------------:|:-----------:|:-------------------------------------------:|------------------:|
| Bagging    |   AB  |  Seq  | 0.0001 |       67.35 |      81.32 |        20        |   model 2   |            bagging1_allbin_0.0001           |    [L](https://drive.google.com/open?id=1J8c1cDiDPIPcFr3-JZN11rwWa8AfCjMw) |
| AdaBoost   |   AB  |  Seq  |   0.01 |       67.08 |      81.93 |        25        |   model 2   |     models_allbin_original_0.01_epoch_30    |    [L](https://drive.google.com/open?id=1V0Lu6qRxrO6RA3LeeeHZJewvwrSvcVZL) |
| AdaBoost   |   AB  |  Indp |   0.01 |       70.59 |      82.12 |        20        |   model 2   |       models_allbin_indp_original_0.01      |   [L](https://drive.google.com/open?id=1ODDG_tuKZZvZBbJLdHsvIoiewcWDu7_3) |
| LogitBoost |   AB  |  Seq  |   0.01 |       62.87 |      82.58 |        30        |   model 2   |  models_allbin_sampling_logit_0.01_epoch_30 |  [L](https://drive.google.com/open?id=1XHeMKAcdjEwW08tLXG3FDfsItqF8jTjr) |
| LogitBoost |   AB  |  Indp |   0.01 |       69.65 |      82.14 |        21        |   model 2   |        models_allbin_indp_logit_0.01        | [L](https://drive.google.com/open?id=1VMR6QmQgaAKh9vHLPe3Ki_VWJG1NWQYM) |
| MildBoost  |   AB  |  Seq  | 0.0001 |       67.88 |      79.40 |        27        |   model 2   | models_allbin_sampling_mild_0.0001_epoch_30 |   [L](https://drive.google.com/open?id=1zDUc69ySbMB9OiQshD2zGgqfPErRXqF9)|
| SAMME      |   AB  |  Indp |  0.001 |       68.72 |      82.04 |        22        |   model 2   |   models_allbin_indp_SAMME_0.001_epoch_30   | SAMME_AB_Indp.txt |
|            |       |       |        |             |            |                  |             |                                             |                   |
| Bagging    |   SB  |  Seq  |  0.001 |       77.87 |      89.12 |        25        |   model 2   |        bagging1_nin_first_model_0.001       |    Bagging_SB.txt |
| AdaBoost   |   SB  |  Seq  |   0.01 |       80.33 |      88.12 |        15        |   model 2   |  models_nin_sampling_original_0.01_epoch_30 |    Ada_SB_Seq.txt |
| LogitBoost |   SB  |  Seq  |  0.001 |       84.23 |       87.9 |        31        |   model 2   |   models_nin_sampling_logit_0.001_epoch_30  |  Logit_SB_Seq.txt |
| MildBoost  |   SB  |  Seq  |  0.001 |       83.68 |      89.00 |        25        |   model 2   |   models_nin_sampling_mild_0.001_epoch_30   |   Mild_SB_Seq.txt |
| MildBoost  |   SB  |  Indp |   0.01 |       80.38 |      87.72 |        23        |   model 2   |          models_nin_indp_mild_0.01          |  Mild_SB_Indp.txt |
| SAMME      |   SB  |  Seq  |  0.001 |        84.5 |      88.83 |        24        |   model 2   |   models_nin_sampling_SAMME_0.001_epoch_30  |  SAMME_SB_Seq.txt |
### Hints

Generally, we have:

:house: 2 different models (you can specify with `--arch allbinnet/semibinnet`)

:hourglass_flowing_sand: 2 different training modes (**independent training**, and **sequential training**)

:gear: 4 different ensemble schemes (**Bagging**, **AdaBoost**, **LogitBoost**, **MildBoost**, and **SAMME**)

<!---
All-Binary Network (AB Model):
    
Independent training:

`python main_bagging_cifar10_train_indp.py --arch allbinnet`

Sequential training:

`python main_bagging_cifar10_train_seq.py --arch allbinnet`

Semi-Binary Network (SB Model):

Independent training:

`python main_bagging_cifar10_train_indp.py --arch semibinnet`

Sequential training:

`python main_bagging_cifar10_train_seq.py --arch semibinnet`


### BENN-Boosting

We offer 3 different boosting options which can be switched by `-b` argument as shown below. You can try more 
boosting strategies as well.

All-Binary Network (AB Model):
    
Independent training:

`python main_boosting_cifar10_train_indp.py --arch allbinnet -b b1/b2/b3`

Sequential training:

`python main_boosting_cifar10_train_seq.py --arch allbinnet -b b1/b2/b3`

Semi-Binary Network (SB Model):

Independent training:

`python main_boosting_cifar10_train_indp.py --arch semibinnet -b b1/b2/b3`

Sequential training:

`python main_boosting_cifar10_train_seq.py --arch semibinnet -b b1/b2/b3`

**Notice:** For AB models, you should get around 79-82% accuracy for 32 ensembles. For SB models, you should get around 87-89% accuracy for 32 ensembles. The single BNN should have around 69-73% and 83-84% accuracy
for AB and SB model respectively.
-->

## Train BENN on ImageNet dataset

The codes and pre-trained models on AlexNet and ResNet-18 will be released soon in near future, please stay tuned.

**Notice:** For AlexNet, you should get around 51-53% (bagging) and 53-55% (boosting) accuracy for 5-6 ensembles. For ResNet-18, you should get around 56-58% (bagging) and 60-62% (boosting) accuracy for 5-6 ensembles. The single BNN 
should have accuracy around 44% and 48% for AlexNet and ResNet-18. Due to overfitting and optimization, you may need to train BENN multiple times and pick the best one. If you
train BENN multiple times, you may end up having a best model that is better than the numbers reported in the paper.


## Train BENN on your own network architecture and dataset

To train BENN for your own application, you can directly reuse the BENN training part of this code. Basically you
only need to modify the files corresponding to the BNN model and the input interface. More details will be provided.

## Acknowledgement

The single BNN training part of this code is mostly written by referencing [XNOR-Net](https://arxiv.org/abs/1603.05279) and Jiecao Yu's [implementation](https://github.com/jiecaoyu/XNOR-Net-PyTorch). Please consider them as well if you 
use our code. Based on our testing, XNOR-Net is the most stable and reliable open source BNN training scheme with product-level codes.

## Check list

- [ ] Release CIFAR-10 Training Code
- [ ] Release CIFAR-10 Pretrained Models
- [ ] Release ImageNet Training Code
- [ ] Release ImageNet Pretrained Models
