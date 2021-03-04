# ViT-CIFAR

Unofficial PyTorch implementation for Vision Transformer[[Dosovitskiy, A.(ICLR'21)]](https://openreview.net/forum?id=YicbFdNTTy) **modified to obtain over 90% accuracy FROM SCRATCH on CIFAR-10 with small number of parameters (= 6.3M)**.

## 1. Quick Start

1. **Install packages**
```sh
$git clone https://github.com/omihub777/ViT-CIFAR.git
$cd ViT-CIFAR/
$bash setup.sh
```

2. **Train ViT on cifar10**

```sh
$python main.py --dataset c10 --label-smoothing --autoaugment
```

* **(Optinal) Train ViT on cifar10 using Comet.ml**  
If you have your [Comet.ml](https://www.comet.ml/) account, this automatically logs experiments by specifying your api key.

```sh
$python main.py --api-key [YOUR COMET API KEY] --dataset c10
```



## 2. Results

|Dataset|Acc.(%)|Time(hh:mm:ss)|
|:--:|:--:|:--:|
|CIFAR-10|**90.92**|02:14:22|
|CIFAR-100|**66.54**|02:14:17|
|SVHN|||

* Number of parameters: 6.3 M
* Device: V100 (single GPU)

## 3. Hyperparams

|Param|Value|
|:--|:--:|
|Epoch|200|
|Batch Size|128|
|Optimizer|Adam|
|Weight Decay|5e-5|
|LR Scheduler|Cosine|
|(Init LR, Last LR)|(1e-3, 1e-5)|
|Warmup|5 epochs|
|Dropout|0.0|
|AutoAugment|ON|
|Label Smoothing|0.1|
|Heads|12|
|Layers|7|
|Hidden|384|
|MLP Hidden|384|

## 4. Side notes
* Longer training gives performance boost.
* More extensive hyperparam search(e.g. LR/Weight Decay/#heads) definitely gives performance gain.