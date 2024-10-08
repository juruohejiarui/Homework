#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
import pickle
import argparse
import sys
import pdb
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from nnUniversal import SelfDataLoader, NNModel
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler

# Train SVM

import torch.nn as nn
import torch.nn.functional as F

import time

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("prefix", type=str)
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--models_num", default=5, type=int)

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, weight=None):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.weight = weight

	def forward(self, inputs, targets):
		ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
		pt = torch.exp(-ce_loss)  # 计算预测的概率
		focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
		return focal_loss

def testModel(model: NNModel, batchLoader : torch.utils.data.DataLoader) -> float :
	accNum, testNum = 0, 0
	for (data, target) in batchLoader :
		size = data.shape[0]
		y : torch.Tensor = model(data.cuda()).cpu()
		y = y.argmax(1)
		target = target.argmax(1)
		for i in range(size) :
			if y[i] == target[i] : accNum += 1
			testNum += 1
	return accNum / testNum

def trainModel(idx : int, model : NNModel, trainBatchLoader : torch.utils.data.DataLoader, validateBatchLoader : torch.utils.data.DataLoader) :
	totalBatch = len(trainBatchLoader)
	criterion = FocalLoss()
	optimizer = torch.optim.SGD(models[i].parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * totalBatch)
	tqdmDesc = tqdm(total=epochs)
	for epoch in range(epochs) :
		model.train()
		for (data, target) in trainBatchLoader :
			x, y = data.cuda(), target.cuda()
			optimizer.zero_grad()
			outputs = model(x)
			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()
			scheduler.step()
			
		model.eval()
		acc = testModel(model, validateBatchLoader)

		tqdmDesc.set_postfix(acc='{:.6f}'.format(acc))
		tqdmDesc.update(1)
		logger.add_scalar(tag=f"accuracy/validate/model{idx}", scalar_value=acc, global_step=epoch + 1)

		acc = testModel(model, trainBatchLoader)
		
		logger.add_scalar(tag=f"accuracy/train/model{idx}", scalar_value=acc, global_step=epoch + 1)

		logger.flush()

if __name__ == '__main__':
	args = parser.parse_args()

	lr = args.lr
	epochs = args.epochs
	batch_size = args.batch_size
	models_num = args.models_num

	logger = SummaryWriter(comment=args.prefix, filename_suffix=".tfevent", flush_secs=1)

	# 1. read all features in one array.
	fread = open(args.list_videos, "r")
	feat_list = []
	# labels are [0-9]
	label_list = []
	# load video names and events in dict
	df_videos_label = {}
	for line in open(args.list_videos).readlines()[1:]:
		video_id, category = line.strip().split(",")
		df_videos_label[video_id] = category


	for line in fread.readlines()[1:]:
		video_id = line.strip().split(",")[0]
		feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
		# for videos with no audio, ignore
		if os.path.exists(feat_filepath):
			feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
			label_list.append(int(df_videos_label[video_id]))

	print("number of samples: %s" % len(feat_list))
	scaler = StandardScaler()

	Y = torch.tensor(np.array(label_list, dtype=np.float32))
	X = torch.tensor(scaler.fit_transform(np.array(feat_list, dtype=np.float32)))

	alpha = 1

	p = [i for i in range(len(label_list))]
	random.shuffle(p)
	validateSize = X.shape[0] // 10
	trainSize = X.shape[0] - validateSize
	X1_train, Y_train = torch.zeros((trainSize, args.feat_dim)), torch.zeros(trainSize)
	X1_valid, Y_valid = torch.zeros((validateSize, args.feat_dim)), torch.zeros(validateSize)
	print("training size: %s, validate size %s" % (trainSize, validateSize))

	for i in range(0, trainSize) :
		X1_train[i], Y_train[i] = X[p[i]], Y[p[i]]
	for i in range(0, validateSize) :
		X1_valid[i], Y_valid[i] = X[p[i + trainSize]], Y[p[i + trainSize]]
	print(X.shape, Y.shape)

	validateDataLoader = SelfDataLoader(X1_valid, Y_valid)
	validateBatchLoader = torch.utils.data.DataLoader(validateDataLoader, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

	trainDataLoader = SelfDataLoader(X1_train, Y_train)
	trainBatchLoader = torch.utils.data.DataLoader(trainDataLoader, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

	# pass array for svm training
	# one-versus-rest multiclass strategy
	models = [NNModel(args.feat_dim).cuda() for i in range(models_num)]
	sample_weights = torch.ones(len(X1_train)) / len(X1_train)
	for i in range(models_num) :	
		trainModel(i, models[i], trainBatchLoader=trainBatchLoader, validateBatchLoader=validateBatchLoader)

	# save trained SVM in output_file
	pickle.dump(models, open(args.output_file, 'wb'))
	pickle.dump(scaler, open("models/scaler", "wb"))
	print('NN trained successfully')

