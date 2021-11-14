#!bin/bash

pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
git clone https://github.com/DeepVoltaire/AutoAugment.git
cd AutoAugment
git checkout 17d718251f25c0d9413bf30f91b523907924f33a
cd ../