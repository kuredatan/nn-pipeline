#coding:utf-8

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle
import image_slicer
from PIL import ImageDraw

from models import autoencoders, ann, kNN, num_classes
from utils import argmax
from data import train_folder, test1_folder, test2_folder

import sys
try:
	idx_os = sys.argv.index("--os")
	os_ = str(sys.argv[idx_os+1])
except:
	os_ = "windows"
tiling = False
## CUDA TODO ADD

#########################
## PROCEDURES          ##
#########################

## Colab
# https://colab.research.google.com/drive/1lU3YKgAHx0NjUE0defnMFuD8YURTc5LO

def aggregation_tiles(labels, type_):
	if (type_ == "mean"):
		label = sum(labels)/len(labels)
		label = label+int(label-int(label) > 0.5)
	else:
		return "Wrong aggregation function: does not exist."
		raise ValueError
	return label

def consider_tiles(names, network, dir_, ntiles=10, type_="mean"):
	labels_ = []
	outputs = None
	for im in names:
		im_tiles = image_slicer.slice(dir_+im, ntiles)
		im_tiles = list(map(lambda x : np.array(x.image), im_tiles))
		im_tensor = torch.tensor(im_tiles).float().permute(0, 3, 1, 2)
		res = network(im_tensor)
		output = torch.sum(res[0], 0).reshape((1, num_classes)) ## TODO
		if (len(np.shape(outputs)) == 0):
			outputs = output
		else:
			outputs = torch.cat((outputs, output), 0)
		label = aggregation_tiles(argmax(res[0]), type_)
		labels_ += [label]
	return outputs, labels_

def train_and_valid(model, epoch, network, loader, val_loader, loss_function, optimizer, filename):
	if (model in ann):
		values = train(epoch, network, loader, loss_function, optimizer)
		values_val = valid(epoch, network, val_loader, loss_function)
	if (model in autoencoders):
		values = train_ae(epoch, network, loader, loss_function, optimizer, filename)
		values_val = valid_ae(epoch, network, val_loader, loss_function, filename)
	if (model in ["knn", "xgboost"]):
		acc_mean, loss_mean, loss_val, acc_val = -1, -1, -1, 0.
		N, N_val = len(loader.dataset), len(val_loader.dataset)
		X, Y = [], []
		for batch_idx, (_, data, target) in tqdm(enumerate(loader)):
			X.append(data[0])
			Y.append(target[0])
		network = network.create(X, Y)
		## Save the network
		str_tree = pickle.dumps(network)
		with open(filename, "w") as f:
			f.write(str_tree)
		values = [-1]*2
		pred, labels = [], []
		for batch_idx, (_, data, target) in tqdm(enumerate(val_loader)):
			pred += network.predict(data)
			labels += list(map(lambda z : int(z.numpy()), target))
		correct_val = accuracy_score(np.array(labels), np.array(pred), normalize=False)
		acc_val = 100. * correct_val / N_val
		print('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
		correct_val, N_val, acc_val))
		values_val = [acc_val/100., -1]
	return values+values_val

def train_ae(epoch, network, loader, loss_function, optimizer, filename):
	network.train()
	loss_mean = 0.
	N = len(loader.dataset)
	for batch_idx, (_, data, target) in tqdm(enumerate(loader)):
		optimizer.zero_grad()
		res = network(data)
		output = res[0]
		if (os_ == "windows"):
			t_loss = loss_function(output, target, reduction="sum")
		else:
			t_loss = loss_function(output, target, size_average=False)
		loss_mean += t_loss.item()
		if (len(res) > 1):
			loss = t_loss + res[1]
			loss_mean += res[1]
		else:
			loss = loss_function(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 50 == 0:
			print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(loader.dataset),
			100. * batch_idx / len(loader), loss.item()))
	X, Y = [], []
	for batch_idx, (_, data, target) in tqdm(enumerate(loader)):
		X.append(network.encode(data)[0])
		Y.append(target[0])
	network = network.create(X, Y)
	## Save the network
	str_tree = pickle.dumps(network)
	with open(filename, "w") as f:
		f.write(str_tree)
	return -1, loss_mean / N

def valid_ae(epoch, network, loader, loss_function, filename):
	network.eval()
	loss_val = 0.
	correct_val = 0
	N = len(loader.dataset)
	with open(filename, "r") as f:
		tree = pickle.load(f)
	for batch_idx, (_, data, target) in tqdm(enumerate(loader)):
		embedding = network.encode(data)[0]
		pred = tree.predict(embedding)
		if (os_=="windows"):
			loss_val += loss_function(torch.tensor(pred).float(), target, reduction="sum").item()
		else:
			loss_val += loss_function(torch.tensor(pred).float(), target, size_average=False).item()
		correct_val += accuracy_score(target.numpy(), pred, normalize=False)
	loss_val /= N
	acc_val = 100. * correct_val / N
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	loss_val, correct_val, N, acc_val))
	return acc_val/100., loss_val	

def train(epoch, network, loader, loss_function, optimizer):
	network.train()
	acc_mean, loss_mean = 0., 0.
	N = len(loader.dataset)
	for batch_idx, (names, data, target) in tqdm(enumerate(loader)):
		optimizer.zero_grad()
		if (tiling):
			output, labels = consider_tiles(names, network, train_folder)
			res = [output]
			acc = accuracy_score(target.numpy(), labels, normalize=False)
		else:
			res = network(data)
			output = res[0]
			acc = accuracy_score(target.numpy(), argmax(output), normalize=False)
		acc_mean += acc
		if (os_ == "windows"):
			t_loss = loss_function(output, target, reduction="sum")
		else:
			t_loss = loss_function(output, target, size_average=False)
		loss_mean += t_loss.item()
		if (len(res) > 1):
			loss = t_loss + res[1]
			loss_mean += res[1]
		else:
			loss = loss_function(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 50 == 0:
			print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
			epoch, batch_idx * len(data), len(loader.dataset),
			100. * batch_idx / len(loader), loss.item(), acc / float(len(target))))
	return acc_mean / N, loss_mean / N

def valid(epoch, network, loader, loss_function):
	network.eval()
	loss_val = 0.
	correct_val = 0
	N = len(loader.dataset)
	for batch_idx, (names, data, target) in tqdm(enumerate(loader)):
		if (tiling):
			output = consider_tiles(names, network, train_folder)
			res = [output]	
		else:
			res = network(data)
			output = res[0]
		if (os_ == "windows"):
			loss_val += loss_function(output, target, reduction="sum")
		else:
			loss_val += loss_function(output, target, size_average=False)
		if (len(res) > 1):
			loss_val += res[1]
		correct_val += accuracy_score(target.numpy(), argmax(output), normalize=False)
	loss_val /= N
	acc_val = 100. * correct_val / N
	print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	loss_val, correct_val, N, acc_val))
	return acc_val/100., loss_val	

def test(model, network, loader, filename):
	pred_classes = {}
	if (model in ann):
		network.eval()
		for (names, data) in tqdm(loader):
			output = network(data)[0]
			pred = argmax(output)
			for i in range(len(names)):
				pred_classes.setdefault(names[i], pred[i])
	if (model in autoencoders):
		try:
			with open(filename, "r") as f:
				tree = pickle.load(f)
		except:
			print("Launch training phase first!")
			print(filename)
			raise ValueError
		for (names, data) in tqdm(loader):
			embedding = network.encode(data)[0]
			pred = tree.predict(embedding)
			for i in range(len(names)):
				pred_classes.setdefault(names[i], pred[i])
	if (model in ["knn", "xgboost"]):
		try:
			with open(filename, "r") as f:
				network = pickle.load(f)
		except:
			print("Launch training phase first!")
			print(filename)
			raise ValueError
		for (names, data) in tqdm(loader):
			pred = network.predict(data)
			for i in range(len(names)):
				pred_classes.setdefault(names[i], pred[i])
	return pred_classes
