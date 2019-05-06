#coding:utf-8

if __name__ == "__main__":

	import torch
	import torchvision
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	import argparse
	import numpy as np

	from data import createDataLoader, createDatasetList
	import models
	from test_models import train_and_valid, test
	from utils import plot_acc_loss, add_to_str_ls, order_results

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch", type=int, help="Batch size.", default=32)
	parser.add_argument("--os", type=str, help="Operating system.", default="windows", choices=["windows", "linux"])
	parser.add_argument("--epochs", type=int, help="Number of epochs.", default=10)
	parser.add_argument("--lr", type=float, help="Learning rate.", default=0.01)
	parser.add_argument("--optim", type=str, help="Optimizing method.", default="sgd", choices=["sgd", "adam"])
	parser.add_argument("--loss", type=str, help="Loss function.", default="crossentropy", choices=["crossentropy"])
	parser.add_argument("--model", type=str, help="Name of the model to train.", default="convnet", choices=["convnet", "lenet", "gao", "jitaree", "vae", "knn", "xgboost"])#"waee"
	parser.add_argument("--load", type=int, help="If set to 1, load pre-trained weights stored in folder ../weights at the specified path with the proper parameters.", default=0, choices=range(2))
	parser.add_argument("--save", type=int, help="If set to 1, save trained weights stored in folder ../weights at the specified path with the proper parameters.", default=0, choices=range(2))
	parser.add_argument("--action", type=str, help="Action to perform.", default="train", choices=["train", "test"])
	parser.add_argument("--K", type=int, help="K number for K-fold cross validation.", default=5)
	parser.add_argument("--p", type=float, help="0 < p < 1 number for simple separation between training and validation set (N total images => int(p*N) images in validation set).", default=0.3)
	parser.add_argument("--shape", type=int, help="If positive, rescale image to a square image of size equal to the argument.", default=0)
	parser.add_argument("--data", type=float, help="0 < q <= 1 percentage of training data to use (speeds up testing).", default=1.)
	parser.add_argument("--name", type=str, help="Give a distinct name for each experiment to avoid confusion.", default="Experiment1")
	args = parser.parse_args()

	## Path to store the results
	results_folder = "../results/"
	filenames = ["accuracy_loss", "result", "knn"]
	filenames = add_to_str_ls(results_folder, filenames, before=True)
	filenames.append("../weights/weights")
	for arg in vars(args):
		add_ = "_"+arg+"="+str(getattr(args, arg))
		filenames = add_to_str_ls(add_, filenames)
	filenames1 = add_to_str_ls(".csv", filenames[:2])
	acc_name, result_name = filenames1
	f_name = filenames[-2]+".txt"
	weights_name = filenames[-1]+".pth"

	## MODEL
	if (args.model == "convnet"):
		network = models.ConvNet(shape=args.shape)
	if (args.model == "lenet"):
		network = models.LeNet(shape=args.shape)
	if (args.model == "gao"):
		network = models.Gao(shape=args.shape)
	if (args.model == "jitaree"):
		network = models.Jitaree(shape=args.shape)
	if (args.model == "vae"):
		network = models.VAE(shape=args.shape, batch_size=args.batch)
	#if (args.model == "waae"):
	#	network = models.WAAE(shape=args.shape)
	if (args.model == "knn"):
		network = models.kNN()
	if (args.model == "xgboost"):
		network = models.XGBoost()
	if (not args.model in ["knn", "xgboost"]):
		if (bool(args.load)):
			network.load_state_dict(torch.load(weights_name))
			network.eval()
		else:
			## Saving initial weights, to avoid data leakage in cross-validation
			torch.save(network.state_dict(), weights_name)

		## OPTIMIZER
		if (args.optim == "sgd"):
			optimizer = optim.SGD(network.parameters(), lr=args.lr)
		if (args.optim == "adam"):
			optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-6)

		## LOSS FUNCTION
		if (args.loss == "crossentropy"):
			loss_function = F.cross_entropy
	else:
		loss_function, optimizer = None, None

	## Load dataset
	if (args.K > 1):
		dataset_list = createDatasetList(args.action, K=args.K, q=args.data)
	else:
		dataset_list = createDatasetList(args.action, K=args.K, p=args.p, q=args.data)

	## acc, loss, acc_val, loss_val
	if (args.action == "train"):
		if (args.K >= 2):
			acc_loss_K = np.zeros((args.epochs, 4, args.K))
		else:
			acc_loss = np.zeros((args.epochs, 4))
	else:
		classes = {}

	# Train
	if (args.action == "train"):
		best_acc = 0.
		## IDX of best training set
		best_idx = None
		## Cross-validation
		if (args.K > 1):
			for i in range(args.K):
				print("Cross-validation = " + str(i+1) + "/" + str(args.K))
				dataset = [dataset_list[i]]
				loader = createDataLoader(dataset, args.batch, True, 0)
				del dataset_list[i]
				val_loader = createDataLoader(dataset_list, args.batch, True, 0)
				dataset_list.insert(i, dataset[0])
				if (not args.model in ["knn", "xgboost"]):
					## Reinitialize weights
					network.load_state_dict(torch.load(weights_name))
					network.eval()
					for epoch in range(1, args.epochs+1):
						acc_loss_K[epoch-1, :, i] = train_and_valid(args.model, epoch, network, loader, val_loader, loss_function, optimizer, filename=f_name)
						## Save at each epoch
						torch.save(network.state_dict(), weights_name[:-4]+"_i="+str(i)+".pth")
				else:
					acc_loss_K[1, :, i] = train_and_valid(args.model, 1, network, loader, val_loader, loss_function, optimizer, filename=f_name)
				if (acc_loss_K[-1,2,i] > best_acc or (not best_idx)):
					best_idx = i
					best_acc = acc_loss_K[-1,2,i]
			acc_loss = acc_loss_K[:,:,best_idx]
			if (not args.model in ["knn", "xgboost"]):
				from subprocess import call
				if (args.os == "windows"):
					cmd = "move"
				else:
					cmd = "mv"
				call(cmd+" ../weights/"+weights_name[:-4]+"_i="+str(best_idx)+".pth ../weights/"+weights_name, shell=True)
		else:
			## Select a random validation set (simple model testing)
			loader = createDataLoader([dataset_list[0]], args.batch, True, 0)
			val_loader = createDataLoader([dataset_list[1]], args.batch, True, 0)
			if (not args.model in ["knn", "xgboost"]):
				network.eval()
				for epoch in range(1, args.epochs+1):
					acc_loss[epoch-1, :] = train_and_valid(args.model, epoch, network, loader, val_loader, loss_function, optimizer, filename=f_name)
					## Save at each epoch
					torch.save(network.state_dict(), weights_name[:-4]+"_training.pth")
			else:
				acc_loss[1, :] = train_and_valid(args.model, 1, network, loader, val_loader, loss_function, optimizer, filename=f_name)
	else:
	# Test
		loader = createDataLoader(dataset_list, args.batch, False, 0)
		pred_classes = test(args.model, network, loader, filename=f_name)
		#classes = {**classes, **pred_classes}
		classes.update(pred_classes)

	## Save files
	if (args.action == "train"):
		np.savetxt(acc_name, acc_loss, delimiter=",")
		## Accuracy/Loss curves
		plot_acc_loss(acc_loss, showit=True)
	else:
		## Order files in results
		res = order_results(classes)
		with open(result_name, "w") as f:
			for im_name, cls in res:
				f.write(im_name+";"+str(cls)+"\n")
