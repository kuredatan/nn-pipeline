#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.neighbors import KDTree
#import xgboost as xgb
import numpy as np
from resnet50 import resnet50

num_classes = 4
autoencoders = ["vae"]#["waae"]
ann = ['gao', 'convnet', 'lenet', 'jitaree']

#########################
## MODELS              ##
#########################

#Source: https://github.com/maitek/waae-pytorch
# class Encoder_WAAE

class kNN:
	def __init__(self, K=1, leaf_size=2):
		self.leaf_size = leaf_size
		self.length = 0
		self.tree = None
		self.labels = None
		self.K = K

	def create(self, x, y):
		x = np.matrix(list(map(lambda y : y.numpy(), x)))
		self.tree = KDTree(x, leaf_size=self.leaf_size)
		self.labels = list(map(lambda z: int(z.numpy()), y))
		self.length = len(x)
		return self

	def predict(self, x):
		pred = []
		## x in batches
		for i in range(len(x)):
			dist, ind = self.tree.query(x[i].numpy().reshape(1, -1), k=self.K)
			pred.append(self.labels[int(ind)])
		return pred

## TODO
class XGBoost:
	def __init__(self, max_depth=2, eta=1, silent=1, objective='binary:logistic', num_round=2):
		self.param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
		self.num_round = 2
		self.bst = None

	def create(self, x, y):
		x = xgb.DMatrix(x.numpy(), label=y)
		self.bst = xgb.train(self.param, x, self.num_round)
		return self

	def predict(self, x):
		x = xgb.DMatrix(x.numpy(), label=None)
		pred = self.bst.predict(x)
		return pred

## Source: https://github.com/yunjey/pytorch-tutorial
## Learn features and feed it to SVM, etc.
class VAE(nn.Module):
	def __init__(self, shape, batch_size, h_dim=40, z_dim=20):
		super(VAE, self).__init__()
		h_dim = 2*shape//3
		z_dim = h_dim//10
		self.shape = shape
		## nn.Linear
		self.conv1 = nn.Conv2d(shape, h_dim, kernel_size=5, padding=2, stride=1)
		self.conv2 = nn.Conv2d(h_dim, z_dim, kernel_size=5, padding=2, stride=1)
		self.conv3 = nn.Conv2d(h_dim, z_dim, kernel_size=5, padding=2, stride=1)
		self.conv4 = nn.Conv2d(z_dim, h_dim, kernel_size=5, padding=2, stride=1)
		self.conv5 = nn.Conv2d(h_dim, batch_size, kernel_size=5, padding=2, stride=1)

	def encode(self, x):
		x = x.permute(0, 2, 3, 1)
		h = F.relu(self.conv1(x))
		return self.conv2(h), self.conv3(h)
    
	def reparameterize(self, mu, log_var):
		std = torch.exp(log_var/2)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z):
		h = F.relu(self.conv4(z))
		return F.sigmoid(self.conv5(h))
    
	def forward(self, x):
		mu, log_var = self.encode(x)
		z = self.reparameterize(mu, log_var)
		x_reconst = self.decode(z).permute(0, 3, 1, 2)
		kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
		return x_reconst, kl_div

#Jitaree, S., Phinyomark, A., Thongnoo, K., Boonyapiphat, P., & Phukpattaranont, P. (2013). 
#Classifying breast cancer regions in microscopic image using texture analysis and neural network

class Jitaree(nn.Module):
	def __init__(self, shape):
		#TODO modify dimensions
		super(Jitaree, self).__init__()
		#lr=0.1 e=0.001 (callback) epochs=100
		if (shape < 1):
			outsize = 70560
		else:
			outsize = ((shape-1)//4-3+2*2)/1+1
			outsize = 10*outsize*outsize
		self.conv1 = nn.Conv2d(3, 4, 5)
		self.conv2 = nn.Conv2d(4, 10, 5)
		self.fc   = nn.Linear(outsize, num_classes)

	def forward(self, x):
		out = x
		out = F.tanh(self.conv1(out))
		out = F.avg_pool2d(out, 2)
		out = F.tanh(self.conv2(out))
		out = F.avg_pool2d(out, 3)
		out = out.view(out.size(0), -1)
		out = F.tanh(self.fc(out))
		out = nn.Softmax(dim=0)(out)
		return [out]

#Gao, Z., Wang, L., Zhou, L., & Zhang, J. (2017). 
#HEp-2 Cell Image Classification With Deep Convolutional Neural Networks

class Gao(nn.Module):
	def __init__(self, shape):
		super(Gao, self).__init__()
		if (shape < 1):
			outsize = 23328
		else:
			outsize = ((shape-1)//32-3+2*2)/1
			outsize = 32*outsize*outsize
		self.conv1 = nn.Conv2d(3, 6, 7)
		self.conv2 = nn.Conv2d(6, 16, 4)
		self.conv3 = nn.Conv2d(16, 32, 3)
		self.fc1   = nn.Linear(outsize, 150)
		self.fc2   = nn.Linear(150, num_classes)

	def forward(self, x):
		out = x
		out = F.relu(self.conv1(out))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 3)
		out = F.relu(self.conv3(out))
		out = F.max_pool2d(out, 3)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = self.fc2(out)
		out = nn.Softmax(dim=0)(out)
		return [out]

#https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

class LeNet(nn.Module):
	def __init__(self, shape):
		super(LeNet, self).__init__()
		if (shape < 1):
			outsize = 11
			outsize = 256032
		else:
			outsize = ((shape-1)//4-5+2*2)/1-1
			outsize = 16*outsize*outsize
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(outsize, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, num_classes)

	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		out = nn.Softmax(dim=0)(out)
		return [out]

#Results were obtained with a standard convolutional neural network 
#(CNN, 3 convolutional layers, 
#each convolutional layer followed by dropout and pooling). 
#The non-weighted accuracy was 75% on 
#the test set. Note that this result is without performing any: 
#- Images pre-processing - 
#Images augmentation (noise, linear transforms or local deformations) 
#- Features computation
#benchmark accuracy: 0.7844

class ConvNet(nn.Module):
	def __init__(self, shape, feat_nb=[3, 16, 32, 64, 100, 64, 32, 16], stride=1, padding=2, filter_size=3, pool_size=2, pool_stride=2):
		super(ConvNet, self).__init__()
		self.nlayers = len(feat_nb)-1
		if (shape < 1):
			shape = 2**self.nlayers*feat_nb[-1]+1
		outsize = ((shape-1)//(2**self.nlayers)-filter_size+2*padding)/stride+1
		for i in range(1, len(feat_nb)):
			layer = nn.Sequential(
				nn.Conv2d(feat_nb[i-1], feat_nb[i], 
					kernel_size=filter_size, 
					padding=padding, stride=stride),
				nn.BatchNorm2d(feat_nb[i]),
				nn.ReLU(),
				nn.Dropout(p=0.3, inplace=True),
				nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride))
			setattr(self, 'conv%d' % i, layer)
		self.fc = nn.Sequential(
				nn.Linear(outsize*outsize*feat_nb[-1], num_classes),
				nn.Softmax(dim=0)
			)
	
	def forward(self, x):
		out = x
		for i in range(1, self.nlayers+1):
				out = getattr(self, 'conv%d' % i)(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return [out]

##TESTS

if __name__ == "__main__":
	from data import createDataLoader, createDatasetList
	from utils import argmax
	from tqdm import tqdm
	from sklearn.metrics import accuracy_score
	network = ConvNet(shape=0)
	dataset = createDatasetList("train")
	loader = createDataLoader(dataset, 10, False, 0)
	batch_idx = 0
	for (names, data, target) in tqdm(loader):
		#print("#batch =", batch_idx)
		output = network(data)[0]
		pred = argmax(output)
		#print(names)
		print(pred)
		print(target.numpy())
		print(accuracy_score(target.numpy(), pred))
		batch_idx += 1
		if (batch_idx > 0):#3):
			break
