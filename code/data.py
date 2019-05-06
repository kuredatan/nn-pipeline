#coding:utf-8

import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import datasets, transforms
import csv
import subprocess as sb
import random

from utils import list_split, load_image

random.seed(123456)

results = "../results/"
data_path = "../data/"
train_folder = data_path + "TrainingSetImagesDir/"
test_folder = data_path + "TestSetImagesDir/"
test1_folder = test_folder + "part_1/"
test2_folder = test_folder + "part_2/"
output_class_file = data_path + "TrainingSet_20aimVO.csv"

import sys
try:
	idx_shape = sys.argv.index("--shape")
	shape = int(sys.argv[idx_shape+1])
except:
	shape = 0
try:
	idx_os = sys.argv.index("--os")
	os_ = str(sys.argv[idx_os+1])
except:
	os_ = "windows"

########################
## DATA LOADING       ##
########################

## FEATURE ENGINEERING
## Co-occurrence/GLCM properties + PCA
## ref: http://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
def feature_build(img):
	from skimage.feature.texture import greycoprops, greycomatrix, local_binary_pattern
	from skimage.color import rgb2gray
	img = np.asarray(rgb2gray(img.numpy()), dtype=np.uint8)
	mat = greycomatrix(img, [1, 2], [0, np.pi/2], levels=4, normed=True, symmetric=True)
	features = []
	if (True):
		features.append(greycoprops(mat, 'contrast'))
		features.append(greycoprops(mat, 'dissimilarity'))
		features.append(greycoprops(mat, 'homogeneity'))
		#features.append(greycoprops(mat, 'energy'))
		#features.append(greycoprops(mat, 'correlation'))
		features = np.concatenate(features)
	else:
		radius = 2
		features = local_binary_pattern(img, 8*radius, radius, method='default') #'ror', 'uniform', 'var'
	feature = features.flatten()
	return torch.tensor(feature).float()

## NORMALIZATION
def normalize(img):
	nimg = img.numpy()
	m, M = np.min(nimg), np.max(nimg)
	nimg -= m
	nimg /= (M-m)
	return torch.tensor(nimg).float()

Normalize = transforms.Normalize([0.]*3, [1.]*3)

normalize_transforms = [transforms.Lambda(normalize)]
feature_build_transforms = []#[transforms.Lambda(feature_build)] #[]

## List of transforms for Data Augmentation
RC = lambda ts : transforms.RandomChoice(ts)
RA = lambda ts : transforms.RandomApply(ts, p=0.7)
RO = lambda ts : transforms.RandomOrder(ts)
HFlip = transforms.RandomHorizontalFlip(p=0.7)
VFlip = transforms.RandomVerticalFlip(p=0.7)
Rotation = transforms.RandomRotation((0, 180), resample=False, expand=False, center=None)
## White noise
noise = transforms.Lambda(lambda img : img+0.01*Variable(torch.randn(img.size())))
Resize = transforms.Lambda(lambda img : transforms.functional.resize(img, (shape, shape)) if (shape > 0) else img)
CJ = transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0.5, hue=0)
## For custom transformations
# transforms.LinearTransformation(transformation_matrix)
# transforms.Lambda(lambd)
data_augmentation = [RC([HFlip, VFlip, Rotation, RA([CJ])])]*2

data_transform_train = transforms.Compose(
	[Resize]+
	[RO(data_augmentation)]+
        [transforms.ToTensor()]+
	[RA([noise])]+
        normalize_transforms+
	feature_build_transforms
)

data_transform_test = transforms.Compose(
	[Resize]+
	[transforms.ToTensor()]+
	normalize_transforms+
	feature_build_transforms
)

class challengeDataset(data.Dataset):
	def __init__(self, path, image_names, transform=None, training=False):
		## List of image names
		self.image_names = image_names
		self.nimages = len(self.image_names)
		self.path = path
		self.transform = transform
		self.targets = {}
		self.training = training
		if (training):
			with open(output_class_file) as f:
				for row in csv.reader(f):
					if (row[0] in image_names):
						self.targets.setdefault(row[0], int(row[1]))

	def __len__(self):
		return self.nimages

	def __getitem__(self, idx):
		key = self.image_names[idx]
		try:
			image = load_image(self.path+key, sz=None)
		except:
			if (not self.training):
				if (self.path == test1_folder):
					image = load_image(test2_folder+key, sz=None)
				else:
					image = load_image(test1_folder+key, sz=None)
			else:
				print("\n\nDid not find image named: '"+self.path+key+"\' in training set folder\n")
		sample = [key]
		if self.transform:
			sample.append(self.transform(image))
		else:
			sample.append(image)
		if (self.training):
			sample.append(self.targets[key])
		return tuple(sample)

## p: percentage of images in validation set when K < 2
## q: percentage of data to keep (between 0 and 1)
def get_image_names(path, K=0, p=0, q=1):
	## List of image names
	try:
		_list = sb.check_output("cd "+path+"; ls", shell=True).splitlines()
	except:
		_list = sb.check_output("cd "+path+" && dir", shell=True).splitlines()
	image_names = list(map(lambda x: x.decode("UTF-8"),_list))
	if (os_ == "windows"):
		image_names = list(map(lambda im : "im_"+im.split(" im_")[-1], image_names))
		image_names = list(filter(lambda im : "im_" in im and ".png" in im, image_names))
	if (q-1<0):
		N = len(image_names)
		image_names = random.sample(image_names, int(q*N))
	image_names = list(filter(lambda im : len(im.split("_"))-1 == 2, image_names))
	random.shuffle(image_names)
	if (K > 1):
		## Then perform cross-validation
		image_names = list_split(image_names, K)
	else:
		if (not p):
			image_names = [image_names]
		else:
			N = len(image_names)
			valid_idx = random.sample(range(N), int(p*N))
			training_set, validation_set = [], []
			for i in range(N):
				if (i in valid_idx):
					validation_set.append(image_names[i])
				else:
					training_set.append(image_names[i])
			image_names = [training_set, validation_set]
	return image_names

im_dict = {"train": [train_folder, data_transform_train], 
"test1": [test1_folder, data_transform_test],
"test2": [test2_folder, data_transform_test]}

def createDatasetList(setname, K=0, p=0, q=1):
	if (setname == "train"):
		folder, transform = im_dict[setname]
		if (K > 1):
			image_names = get_image_names(folder, K, q=q)
		else:
			## Select a random validation set
			image_names = get_image_names(folder, p=p, q=q)
		sets = [challengeDataset(folder, im_ls, transform, training=True) for im_ls in image_names]
	else:
		q0 = 1 if (q-1==0) else 0.5*q
		folder1, transform = im_dict["test1"]
		image_names = get_image_names(folder1, 0, q=q0)
		folder2, transform = im_dict["test2"]
		image_names += get_image_names(folder2, 0, q=q0)
		sets = [challengeDataset(folder1, image_names[0], transform)]
		sets += [challengeDataset(folder2, image_names[1], transform)]
	return sets

def createDataLoader(sets, batchsize, shuffle, num_workers):
	if (len(sets) > 1):
		dataset = data.ConcatDataset(sets)
	else:
		dataset = sets[0]
	return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)

### TESTS

if __name__ == "__main__":
	if (False):
		for x in [["train", 5], ["train", 8], ["test1", 0], ["test2", 0], ["train", 1]]:
			print("*"*15)
			print("Arguments = folder:", x[0], "K:", x[1])
			folder, _, _ = im_dict[x[0]]
			ls = get_image_names(folder, x[1])
			print("Length of list = ", len(ls))
			print("Lengths inside list = ", list(map(len, ls)))
		####################################
		print("*"*15)
		K = 5
		ls = get_image_names(train_folder, K)[0]
		print("path:", train_folder, "K:", K)
		datasets = createDatasetList("train", K=K)
		dl = data.DataLoader(datasets, batch_size=64, shuffle=True, num_workers=0)
		print(datasets[0].image_names)
		print(datasets[0].targets)
	####################################
	## Testing the data augmentation pipeline
	import matplotlib.pyplot as plt
	dataset_list = createDatasetList("train", K=0, p=0, q=0.1)
	loader = createDataLoader(dataset_list, 1, True, 0)
	n = 5
	plt.figure(figsize=(5, 5*n))
	for batch_idx, (name, data, target) in enumerate(loader):
		if (len(data.shape) == 4):
			data = np.transpose(data[0,:,:,:], (1, 2, 0))
		plt.subplot(str(n)+'1'+str(batch_idx+1))
		plt.axis('off')
		plt.imshow(data)
		if (batch_idx >= n-1):
			break
	plt.show()
