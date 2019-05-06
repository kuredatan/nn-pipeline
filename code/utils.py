#coding: utf-8

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import sample
import csv
from tqdm import tqdm
import os

ntotal_patients = 61
# wc -l ../data/TrainingSet_20aimVO.csv -1 (header)
nrow = 9446 

results = "../results/"
data_path = "../data/"
train_folder = data_path + "TrainingSetImagesDir/"
test_folder = data_path + "TestSetImagesDir/"
test1_folder = test_folder + "part_1/"
test2_folder = test_folder + "part_2/"
output_class_file = data_path + "TrainingSet_20aimVO.csv"
order_file = data_path + "test_data_order.csv"

#' @param classes dictionary (key=image name, value=predicted class)
#' @return res a (N+1)-sized list of 2-sized lists (one list per line in the CSV file) 
#' (where N was the number of elements in classes)
#' ordered according to the order given by the file order_file
def order_results(classes):
	lst = []
	warn = []
	N = 0
	with open(order_file) as f:
		header = False
		for row in csv.reader(f):
			if (not header):
				header = True
				lst.append(["image_filename", "class_number"])
				continue
			N += 1
			cls = classes.get(row[0], -1)
			lst.append([row[0], cls])
	return lst

## Prediction of output of softmax layer of neural network
## Returns a Numpy array
def argmax(output):
	return output.data.max(1, keepdim=True)[1].numpy().T[0]

## Adds string add_ to all strings in list s
def add_to_str_ls(add_, s, before=False):
	if (before):
		f = lambda x : add_+x
	else:
		f = lambda x : x+add_
	return list(map(f, s))

## Cut list a in n similarly-sized chunks
## source: https://stackoverflow.com/questions/2130016/
def list_split(a, n):
	k, m = divmod(len(a), n)
	return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#' @param acc_loss numpy matrix of size nepochs x 4 (acc, loss, acc_val, loss_val)
#' @return accuracy/loss curves
def plot_acc_loss(acc_loss, model="", showit=False):
	nepochs = np.shape(acc_loss)[0]
	epochs = range(1, nepochs+1)
	acc, loss, acc_val, loss_val = [acc_loss[:,i] for i in range(4)]
	plt.figure(figsize=(15, 15))
	plt.subplot('121')
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, acc_val, 'b', label='Validation acc')
	plt.xticks(epochs)
	plt.title('Training and validation accuracy for model ' + model)
	plt.legend()
	plt.subplot('122')
	plt.plot(epochs, loss, 'ro', label='Training loss')
	plt.plot(epochs, loss_val, 'r', label='Validation loss')
	plt.xticks(epochs)
	plt.title('Training and validation loss for model ' + model)
	plt.legend()
	plt.savefig(results+"acc_loss_curve.png", bbox_inches="tight")
	if (showit):
		plt.show()

def load_image(name, sz=32):
	if (not sz):
		return Image.open(name)
	return imresize(Image.open(name), (sz, sz, 3))

#' @param classes array of integers of classes to be plotted (in [0, 1, 2, 3])
#' @param npatients_per_class number of different patients per class 
#' @param ntimes_per_patient number of times images from the same patient appear 
#' @return plot of nclasses x npatients_per_class x ntimes_per_patient
#' 	randomly chosen images from training data
#' 	with the associated patient IDs and output classes
def plot_classes_patients(classes=range(4), npatients_per_class=3, ntimes_per_patient=3, showit=False):
	assert all([c in range(4) for c in classes]), "Absent class"
	assert npatients_per_class>0, "No patient in class"
	assert ntimes_per_patient>0, "Each patient must appear at least once"
	nclasses = len(classes)
	output_classes, patient_names, images_names = [], [], []
	with open(output_class_file) as f:
		header = False
		for row in csv.reader(f):
			if (not header):
				header = True
				continue
			output_classes.append(int(row[1]))
			patient_names.append(int(row[0].split("_")[-1].split("png")[0][:-1]))
			images_names.append(train_folder + row[0])
	## Load preprocessed input
	patient_ = []
	for i in range(ntotal_patients):
		lst = []
		for j in range(nrow):
			if (patient_names[j] == i):
				lst.append((j, output_classes[j], i))
		patient_.append(lst)
	class_ = []
	for c in classes:
		lst = []
		for j in range(ntotal_patients):
			ls = [label[1] for label in patient_[j]]
			## ONE PATIENT DOES NOT IMPLY ONE SINGLE CLASS
			#print(len(list(set(ls))) <= 1)
			if (c in ls):
				lst.append(j)
		class_.append(lst)
	## CLASSES ARE NOT BALANCED IN TRAINING DATA
	#for i in range(nclasses):
	#	print("#patients in class " + str(classes[i]) + " (training data) = " + str(len(class_[i])))
	plotted_images_names = []
	for i in range(nclasses):
		idx_patients = sample(class_[i], npatients_per_class)
		idx_samples = [sample([label[0] for label in patient_[idx]], ntimes_per_patient) for idx in idx_patients]
		idx_samples = [y for x in idx_samples for y in x]		
		plotted_images_names.append(idx_samples)
	m = npatients_per_class*ntimes_per_patient
	idx = 1
	plt.figure(figsize=(20*m,10*nclasses))
	for c in tqdm(range(nclasses)):
		for u in tqdm(range(len(plotted_images_names[c]))):
			plt.subplot(m, nclasses, idx)
			idx += 1
			i = plotted_images_names[c][u]
			patient, cls, im = patient_names[i], output_classes[i], load_image(images_names[i])
			plt.imshow(im)
			plt.axis('off')
			plt.text(0, -1, 'Patient: ' + str(patient) + ' Class: ' + str(cls))
	class_str = ""
	for c in classes:
		class_str += str(c)
	plt.savefig("../images/nclasses="+class_str+"_npat="+str(npatients_per_class)+"_ntimes="+str(ntimes_per_patient)+"", bbox_inches="tight")
	if (showit):
		plt.show()

### TESTS

if __name__ == "__main__":
	plot_classes_patients(classes=[0], npatients_per_class=5, ntimes_per_patient=2)
	plot_classes_patients(classes=[1], npatients_per_class=5, ntimes_per_patient=2)
	plot_classes_patients(classes=[2], npatients_per_class=5, ntimes_per_patient=2)
	plot_classes_patients(classes=[3], npatients_per_class=5, ntimes_per_patient=2)
