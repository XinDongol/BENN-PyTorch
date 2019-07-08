# BENN
Codes for Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?

CVPR 2019 [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.pdf)

If using the code, please cite our paper: [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:H6FM7lSqW_EJ:scholar.google.com/&output=citation&scisdr=CgVN_q1kELej_P-D-dE:AAGBfm0AAAAAXSKG4dFrBBn2csdOlj4BwHlxi6htCY4M&scisig=AAGBfm0AAAAAXSKG4WJt5ECQOvH5NEru1ApeOvNgHKXL&scisf=4&ct=citation&cd=-1&hl=en)

**Notice:** As mentioned in the paper (Section 7) we are aware of the overfitting problem caused by the ensemble technique. If retraining the models, they should basically match the results shown in the paper, but could be slightly higher or lower due to random initialization and overfitting.

## CheckList

- [ ] Release CIFAR-10 Training Code
- [ ] Release ImageNet Training Code
- [ ] Release CIFAR-10 Pretrained Models
- [ ] Release ImageNet Pretrained Models
- [ ] Release FPGA Implementation

## Train BENN on CIFAR-10 Dataset

A customized Network-In-Network (NIN) model is used. Please see paper for architecture details.

### BENN-Bagging

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

**Notice:** For AB models, you should get around 79-82% accuracy for 32 ensembles. For SB models, you should get around 87-89% accuracy for 32 ensembles.


## Train BENN on ImageNet Dataset

AlexNet and ResNet-18 are used for verifying BENN on ImageNet (2012).

### BENN-Bagging

Semi-Binary Network (SB Model), Independent training:

`python main_bagging_imagenet_train_indp.py --arch AlexNet`

`python main_bagging_imagenet_train_indp.py --arch resnet`


### BENN-Boosting

Semi-Binary Network (SB Model), Independent training:

`python main_boosting_imagenet_train_seq.py --arch AlexNet -b b1/b2/b3`

`python main_boosting_imagenet_train_seq.py --arch resnet -b b1/b2/b3`

**Notice:** For AlexNet, you should get around 51-53% (bagging) and 53-55% (boosting) accuracy for 5-6 ensembles. For ResNet-18, you should get around 56-58% (bagging) and 60-62% (boosting) accuracy for 5-6 ensembles. Due to overfitting and optimization, you may need to train BENN multiple times and pick the best one. If you
train BENN multiple times, you may end up having a best model that is better than the numbers reported in the paper.


