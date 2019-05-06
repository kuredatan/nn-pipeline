#coding:utf-8

## From Gul Varol and Ignaccio Rocco HW

# Install PyTorch (http://pytorch.org/)
from os import path
import subprocess as sb
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

#!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
sb.call("pip install -q http://download.pytorch.org/whl/"+accelerator+"/torch-0.4.0-"+platform+"-linux_x86_64.whl torchvision", shell=True)
import torch
print(torch.__version__)
print(torch.cuda.is_available())
