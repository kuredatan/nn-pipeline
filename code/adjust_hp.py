#coding: utf-8

import subprocess as sb

## Grid Search for learning rate
for lr in range(1, 100, 10):
	sb.call("python3 main.py --batch 32 --epochs 10 --lr "+str(lr/100.)+" --load 0 --save 0 --action train --K 5 --p 0.3 --shape 100 --data 0.6 --name Experiment2", shell=True)

## Ensembling
