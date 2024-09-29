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
from nnUniversal import CNNDataLoader, CNNModel
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Train SVM

import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--epochs", default=200)
parser.add_argument("prefix", type=str)
parser.add_argument("--mfccDir", default='data/mfcc.tgz/mfcc/')

batchSize = 30

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

if __name__ == '__main__':
	args = parser.parse_args()
	logger = SummaryWriter(comment=args.prefix, filename_suffix=".tfevent")

	# 1. read all features in one array.
	fread = open(args.list_videos, "r")
	feat_list, featIdx_list = [], []
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
			featIdx_list.append(int(video_id[2: ]))
			label_list.append(int(df_videos_label[video_id]))
	files = [None] * 8000
	Y = torch.tensor(np.array(label_list, dtype=np.float32))
	X1 = torch.tensor(np.array(feat_list, dtype=np.float32))
	X2 = torch.tensor(np.array(featIdx_list, dtype=np.int32))
	for file in os.listdir(args.mfccDir) :
		path = os.path.join(args.mfccDir, file)
		idx = int(file[2 : 10])
		files[idx] = torch.tensor(np.array(pd.read_csv(path, sep=';', dtype="float")))
	

	p = [i for i in range(len(label_list))]
	random.shuffle(p)
	validateSize = X1.shape[0] // 10
	trainSize = X1.shape[0] - validateSize
	X1_train, X2_train, Y_train = torch.zeros((trainSize, args.feat_dim)), torch.zeros(trainSize), torch.zeros(trainSize)
	X1_valid, X2_valid, Y_valid = torch.zeros((validateSize, args.feat_dim)), torch.zeros(validateSize), torch.zeros(validateSize)
	print("training size: %s, validate size %s" % (trainSize, validateSize))
	corTrain = [0] * 10
	corValid = [0] * 10
	for i in range(0, trainSize) :
		X1_train[i], X2_train[i], Y_train[i] = X1[p[i]], X2[p[i]], Y[p[i]]
		corTrain[Y[p[i]].long()] += 1
	for i in range(0, validateSize) :
		X1_valid[i], X2_valid[i], Y_valid[i] = X1[p[i + trainSize]], X2[p[i + trainSize]], Y[p[i + trainSize]]
		corValid[Y[p[i + trainSize]].long()] += 1

	trainDataLoader = CNNDataLoader(X1_train, X2_train, Y_train, files, 1020, 40)
	trainLoader = torch.utils.data.DataLoader(trainDataLoader, batch_size=batchSize, shuffle=True, drop_last=False, num_workers=2)
	validateDataLoader = CNNDataLoader(X1_valid, X2_valid, Y_valid, files, 1020, 40)
	lr = args.lr
	epochs = args.epochs

	# pass array for svm training
	# one-versus-rest multiclass strategy
	model = CNNModel(10, 1020, 40, args.feat_dim)
	model = model.to('cuda')
	totalBatch = len(trainLoader)
	NUM_BATCH_WARM_UP = totalBatch * 5
	criterion = FocalLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * totalBatch, 1e-6)
	tqdmDesc = tqdm(total=epochs)
	
	for epoch in range(epochs) :
		model.train()
		for batch_idx, (data, target) in enumerate(trainLoader):
			x, y = (data[0].cuda(), data[1].cuda()), target.cuda()
			optimizer.zero_grad()
			outputs = model(x[0], x[1], True)
			loss = criterion(outputs, y)
			loss.backward()
			optimizer.step()
			scheduler.step()

		model.eval()
		accCnt = 0
		ansCnt = [0] * 10
		for (x1, x2), y in validateDataLoader :
			modelY = model(x1.cuda(), x2.cuda(), False).cpu()
			ansCnt[torch.argmax(modelY)] += 1
			if torch.argmax(modelY) == torch.argmax(y) : accCnt += 1
		tqdmDesc.set_postfix(acc='{:.6f}'.format(accCnt / validateSize))
		tqdmDesc.update(1)
		logger.add_scalar(tag="accuracy/validate", scalar_value=accCnt / validateSize , global_step=epoch + 1)
		for i in range(10) :
			logger.add_scalar(tag=f"freq/validate/{i}", scalar_value=ansCnt[i] / validateSize, global_step=epoch + 1)
			# logger.add_scalar(tag=f"correct/validate/{i}", scalar_value=corValid[i] / validateSize, global_step=epoch + 1)


		accCnt = 0
		ansCnt = [0] * 10
		for data, target in trainLoader:
			x, y = (data[0].cuda(), data[1].cuda()), target.cuda()
			modelY = model(x[0], x[1], True)
			for i in range(y.shape[0]) :
				ansCnt[torch.argmax(modelY[i])] += 1
				if torch.argmax(modelY[i]) == torch.argmax(target[i]) : accCnt += 1
		logger.add_scalar(tag="accuracy/train", scalar_value=accCnt / trainSize, global_step=epoch + 1)
		for i in range(10) :
			logger.add_scalar(tag=f"freq/train/{i}", scalar_value=ansCnt[i] / trainSize, global_step=epoch + 1)
			# logger.add_scalar(tag=f"correct/train/{i}", scalar_value=corTrain[i] / trainSize, global_step=epoch + 1)

		logger.flush()
	
	accCnt = 0
	for (x1, x2), y in validateDataLoader :
		modelY = model(x1.cuda(), x2.cuda(), False).cpu()
		if torch.argmax(modelY) == torch.argmax(y) : accCnt += 1
	print(f"train set accuracy: {accCnt / validateSize}")
	
	
	# save trained SVM in output_file
	pickle.dump(model, open(args.output_file, 'wb'))
	print('One-versus-rest multi-class SVM trained successfully')

