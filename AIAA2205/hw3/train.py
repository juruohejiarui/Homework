import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
import dataset

import models
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--momentum", type=float, default=0.78)
parser.add_argument("--log_suffix", type=str, default="def")

transforms = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

if __name__ == "__main__" :
	args = parser.parse_args()
	lr = args.lr
	model_name = args.model_name
	optimizer_name = args.optimizer
	epochs = args.epochs
	weight_decay = args.weight_decay
	momentum = args.momentum

	log = SummaryWriter(f"run/{model_name}-{args.log_suffix}", flush_secs=1)

	oridf = dataset.loadDf("data/trainval.csv")
	p = [i for i in range(len(oridf))]
	random.shuffle(p)
	df = [oridf.iloc[i, :] for i in p]

	validSize = len(p) // 8
	trainSize = len(p) - validSize
	print(f"trainSize:{trainSize} validSize:{validSize}")

	val_dataset = dataset.MyDataset('data/hw3_16fpv', df[trainSize : ], stage="val", transform=transforms)
	train_dataset = dataset.MyDataset('data/hw3_16fpv', df[: trainSize], stage="train", transform=transforms)

	print('dataset loaded')
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5, pin_memory=True, drop_last=False)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=5, pin_memory=True)

	print('train ', len(train_loader))
	print('val ', len(val_loader))

	# model = models.VideoResNet(num_classes=10).cuda()
	model = models.VGGLSTM(num_classes=10).cuda()
	if optimizer_name == "adam" :
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
		scheduler = None
		print("optimizer : adam")
	else :
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader), 1e-7)
		print("optimizer: SGD")
	criterion = models.FocalLoss()

	print("model loaded")

	best_acc_train = 0
	best_acc_val = 0
	counter = 0  # Initialize counter to track epochs since last improvement

	print('start training')
	for epoch in tqdm(range(epochs)):
		start_time = time.time() 
		running_loss_train = 0.0
		running_loss_val = 0.0
		correct_train = 0
		total_train = 0
		correct_val = 0
		total_val = 0

		model.train()  # Set the model to train mode
		for inputs, labels in train_loader:
			inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()
			outputs = model(inputs)
			loss_train = criterion(outputs, labels)
			loss_train.backward()

			running_loss_train += loss_train.item()
			predicted_train = torch.argmax(outputs, 1)
			total_train += labels.size(0)
			correct_train += (predicted_train == torch.argmax(labels, 1)).sum().item()

			optimizer.step()
			if scheduler != None : scheduler.step()



		with torch.no_grad():
			model.eval()  # Set the model to evaluation mode
			for val_inputs, val_labels in val_loader:
				val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
				val_outputs = model(val_inputs)
				loss_val = criterion(val_outputs, val_labels)
				running_loss_val += loss_val.item()
				predicted_val = torch.argmax(val_outputs, 1)
				total_val += val_labels.size(0)
				correct_val += (predicted_val == torch.argmax(val_labels, 1)).sum().item()
		
		acc_train = correct_train / total_train
		acc_val = correct_val / total_val
		
		# print(f'acc train {acc_train}, {correct_train}/{total_train}')
		# print(f'acc val	{acc_val}, {correct_val}/{total_val}')

		log.add_scalar("accuracy/train", correct_train / total_train, epoch + 1)
		log.add_scalar("accuracy/valid", correct_val / total_val, epoch + 1)
		log.add_scalar("loss/train", running_loss_train, epoch + 1)
		log.add_scalar("loss/valid", running_loss_val, epoch + 1)

		if acc_train > best_acc_train:
			best_acc_train = acc_train
			torch.save(model.state_dict(), f'models/{model_name}_best_train.pth')

		if acc_val > best_acc_val:
			best_acc_val = acc_val
			torch.save(model.state_dict(), f'models/{model_name}_best_val.pth')
	
		torch.save(model.state_dict(), f'models/{model_name}.pth')