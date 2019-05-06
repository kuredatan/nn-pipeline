#!/bin/bash

easy_install pip
pip install -r requirements.txt
pip install scipy
pip install tqdm
pip install scikit-image
pip install scikit-learn
pip install xgboost
pip install image_slicer
python challenge-ens-2019/code/install_pytorch.py
