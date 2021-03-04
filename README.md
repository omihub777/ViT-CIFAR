# ViT-CIFAR

Unofficial PyTorch implementation for Vision Transformer[[Dosovitskiy, A.(ICLR'21)]](https://openreview.net/forum?id=YicbFdNTTy) **modified to obtain over 90% accuracy on CIFAR-10 with small number of parameters (= 6.3M)**.

## 1. Quick Start

1. **Install packages**
```sh
$git clone https://github.com/omihub777/ViT-CIFAR.git
$cd ViT-CIFAR/
$bash setup.sh
```

2. **Train vit on cifar10**

```sh
$python main.py --dataset c10 --label-smoothing --autoaugment
```

* **(Optinal) Train vit on cifar10 using Comet.ml**
If you have your [Comet.ml](https://www.comet.ml/) account, this automatically logs experiments by specifying your api key.

```sh
$python main.py --api-key [YOUR COMET API KEY] --dataset c10
```



## 2. Results

|Dataset|Acc.(%)|
|:--:|:--:|
|CIFAR-10|**91.01**|
|CIFAR-100||
|SVHN||

* Number of parameters: 6.3 M

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