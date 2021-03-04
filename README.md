# ViT-CIFAR

Unofficial PyTorch implementation for Vision Transformer[[Dosovitskiy, A.(ICLR'21)]](https://openreview.net/forum?id=YicbFdNTTy) **modified to obtain over 90% accuracy on CIFAR-10 in small number of parameters**.

## Quick Start

1. **Install packages**
```sh
$bash setup.sh
```

2. **Train vit on cifar10**

```sh
$python main.py --dataset c10 \
--label-smoothing --autoaugment
```

* **(Optinal) Train vit on cifar10 using Comet.ml**
If you have your [Comet.ml](https://www.comet.ml/) account, this automatically logs experiments by specifing your api key.

```sh
$python main.py --api-key [YOUR COMET API KEY] \
--dataset c10
```



## Result

|Dataset|Acc.(%)|#Params|
|:--:|:--:|:--:|
|CIFAR-10|**91.01**|6.3M|
|CIFAR-100|||

## Hyperparams

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
|Hidden Size|384|