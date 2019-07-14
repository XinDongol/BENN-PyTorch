# BENN
Codes for Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?

CVPR 2019 [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.pdf)

:octocat: If using the code, please cite our paper: [BibTex](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.html)

If you have any question related to the codes or models, please open an issue. If you have general questions about principle of BENN or have any further idea of improving it, please contact us by email: shz338@eng.ucsd.edu, xindong@g.harvard.edu. Please, no commercial use before getting permission from authors.

**Notice:** As mentioned in the paper (Section 7) we are aware of the overfitting problem caused by the ensemble technique. If retraining the models, they should basically match the results shown in the paper as well as here, but could be either slightly higher or lower due to random initialization, epoch selection, overfitting, etc. If you have a good idea of how to resolve the overfitting issue of ensemble methods, please contact the authors and we can further improve BENN.

## Train BENN on CIFAR-10 dataset

A customized Network-In-Network (NIN) model is used. Please see paper for architecture details.

| Ensemble   | Model | Train |     LR | BNN (start) | BENN (end) | Overfitting from | Best Voting |               Models Directory              |              Logs |
|------------|:-----:|:-----:|-------:|------------:|-----------:|:----------------:|:-----------:|:-------------------------------------------:|------------------:|
| Bagging    |   AB  |  Seq  | 0.0001 |       67.35 |      81.32 |        20        |   Soft Max Vote   |            [models_bagging_AB](https://drive.google.com/open?id=1PBppowCQOWr7k4jPuyRoSbSdtQBmcP9N)           |    [L](https://drive.google.com/open?id=1J8c1cDiDPIPcFr3-JZN11rwWa8AfCjMw) |
| BoostA   |   AB  |  Seq  |   0.01 |       67.08 |      81.93 |        25        |   Soft Max Vote   |     [models_boostA_AB_seq](https://drive.google.com/open?id=181YNIyFWlkMH30xTWxy3A0Pm5hBbBAnD)    |    [L](https://drive.google.com/open?id=1V0Lu6qRxrO6RA3LeeeHZJewvwrSvcVZL) |
| BoostA   |   AB  |  Indp |   0.01 |       70.59 |      82.12 |        20        |   Soft Max Vote   |       [models_boostA_AB_indp](https://drive.google.com/open?id=17mwH0zc_ojissgNlu7im1-W2hSsb7nDo)      |   [L](https://drive.google.com/open?id=1ODDG_tuKZZvZBbJLdHsvIoiewcWDu7_3) |
| BoostB |   AB  |  Seq  |   0.01 |       62.87 |      82.58 |        30        |   Soft Max Vote   |  [models_boostB_AB_seq](https://drive.google.com/open?id=1tonq6We35NVH6xEr9FpE-En7l2o4Q-1z) |  [L](https://drive.google.com/open?id=1XHeMKAcdjEwW08tLXG3FDfsItqF8jTjr) |
| BoostB |   AB  |  Indp |   0.01 |       69.65 |      82.13 |        21        |   Soft Max Vote   |        [models_boostB_AB_indp](https://drive.google.com/open?id=1gSe53ExXjxIxJpqw6j19HXfi08kmzQsQ)        | [L](https://drive.google.com/open?id=1VMR6QmQgaAKh9vHLPe3Ki_VWJG1NWQYM) |
| BoostC  |   AB  |  Seq  | 0.0001 |       67.88 |      79.40 |        27        |   Soft Max Vote   | [models_boostC_AB_seq](https://drive.google.com/open?id=1iQSflQg5P3HcD5CGIU_Rz-mGQJv6zx22) |   [L](https://drive.google.com/open?id=1zDUc69ySbMB9OiQshD2zGgqfPErRXqF9)|
| BoostD      |   AB  |  Indp |  0.001 |       68.72 |      82.04 |        22        |   Soft Max Vote   |   [models_boostD_AB_indp](https://drive.google.com/open?id=1uuGlTBsJ6vTIIe620otnsHcL6nwzJBqS)   | [L](https://drive.google.com/open?id=1StHIYfDdiyVu2XCrH4xALjswC94X07ML) |
|            |       |       |        |             |            |                  |             |                                             |                   |
| Bagging    |   SB  |  Seq  |  0.001 |       77.87 |      89.12 |        25        |   Soft Max Vote   |        [models_bagging_SB](https://drive.google.com/open?id=1LXAaqjn4w3BzCAqxkPumeNXxebXOgu7P)       |    [L](https://drive.google.com/open?id=1jMm_4ICENzA2fs-wWePBroGTtCtS7NIG) |
| BoostA   |   SB  |  Seq  |   0.01 |       80.33 |      88.12 |        15        |   Soft Max Vote   |  [models_boostA_SB_seq](https://drive.google.com/open?id=1pDtaywChknD8KaaNxUZDIOlWmmOKTrHq) |    [L](https://drive.google.com/open?id=1YR9kvKhWjbx-pO4pP8743TpUShZi_F7c) |
| BoostB |   SB  |  Seq  |  0.001 |       84.23 |       87.9 |        31        |   Soft Max Vote   |   [models_boostB_SB_seq](https://drive.google.com/open?id=11bx-iEIpKLf7cF27OpXNMH61UD-6n1bN)  |  [L](https://drive.google.com/open?id=1crSzVBfJ-C5bh27wOb4cH_ZHZQv5c41p) |
| BoostC  |   SB  |  Seq  |  0.001 |       83.68 |      89.00 |        25        |   Soft Max Vote   |   [models_boostC_SB_seq](https://drive.google.com/open?id=1_myF7GbhOJGcb64g0z6xzuFt4s7ywmiu)   |   [L](https://drive.google.com/open?id=1agfjAdntT0llIMt9ERFehHBlOS_Ef6_A) |
| BoostC  |   SB  |  Indp |   0.01 |       80.38 |      87.72 |        23        |   Soft Max Vote   |          [models_boostC_SB_indp](https://drive.google.com/open?id=1dNBKR88sSA2-R98yc_3TmFAxUZaED3S8)          |  [L](https://drive.google.com/open?id=1mUPoEeFDrkpl0GX77qzGjY1NNO9KiO1b) |
| BoostD      |   SB  |  Seq  |  0.001 |        84.5 |      88.83 |        24        |   Soft Max Vote   |   [models_boostD_SB_seq](https://drive.google.com/open?id=1mhnJFbs3knOsSr-B-YAt1S5v-j29ksJJ)  |  [L](https://drive.google.com/open?id=1CcnXGiN6cWePt-S-uf1xjOtE_qLlpUE6) |
### Hints

Generally, we have:

:house: 2 different models (you can specify with `--arch allbinnet/nin`), corresponding to AB and SB models in the paper

:hourglass_flowing_sand: 2 different training modes (**independent training**, and **sequential training**)

:gear: 5 different ensemble schemes (**Bagging**, **Boost A**, **Boost B**, **Boost C**, and **Boost D**)

:bar_chart: 2 voting strategies (**hard majority vote**, **soft max vote**)

### Retrain models

For example:

`$ python main_bagging_SB.py --epochs 0 --retrain_epochs 100 --root_dir PATH/TO/YOUR/models_bagging_SB/`

### Test pre-trained models

First download the models from the links above, then run the corresponding python script to test pre-trained models and you should get the exact same numbers comparing with our logs above. For example:

```$ python main_bagging_SB.py --epochs 0 --retrain_epochs 0 --root_dir PATH/TO/YOUR/DOWNLOADED/models_bagging_SB/```

**Notice:** For AB models, you should get around 79-82% accuracy for 32 ensembles. For SB models, you should get around 87-89% accuracy for 32 ensembles (usually 15-20 is a reasonable choice due to overfitting). The single BNN should have around 69-73% and 83-84% accuracy for AB and SB model respectively.

## Train BENN on ImageNet dataset

The codes and pre-trained models on AlexNet and ResNet-18 will be released soon in near future, please stay tuned. We are currently studying overfitting issue and testing the stability of the gain.

**Notice:** For AlexNet, you should get around 50-53% (bagging) and 52-55% (boosting) accuracy for 5-6 ensembles. For ResNet-18, you should get around 56-59% (bagging) and 59-62% (boosting) accuracy for 5-6 ensembles. The single BNN 
should have accuracy around 44% and 48% for AlexNet and ResNet-18. Be sure to use SB model with independent training, and make sure each BNN is well converged before ensemble. Due to overfitting and optimization instability as observed in Section 6.2 from the paper, you may want to train BENN multiple times and pick the best one.


## Train BENN on your own network architecture and dataset

To train BENN for your own application, you can directly reuse the BENN training part of this code. More details will be provided. If you successfully train BENN on some new applications with new architectures and achieve satisfying performance, please contact the authors and we will add a link here.

## Acknowledgement

The single BNN training part of this code is mostly written by referencing [XNOR-Net](https://arxiv.org/abs/1603.05279) and Jiecao Yu's [implementation](https://github.com/jiecaoyu/XNOR-Net-PyTorch). Please consider them as well if you 
use our code. Based on our testing, XNOR-Net is the most stable and reliable open source BNN training scheme with product-level codes.

## Check list

- [x] Release CIFAR-10 Training Code
- [x] Release CIFAR-10 Pretrained Models
- [ ] Release ImageNet Training Code
- [ ] Release ImageNet Pretrained Models
