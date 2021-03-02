#!bin/bash

pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
git clone https://github.com/DeepVoltaire/AutoAugment.git