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
from nnUniversal import SelfDataLoader, NNModel
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

# Train SVM

import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--lr", default=0.005)
parser.add_argument("--epochs", default=300)
parser.add_argument("prefix", type=str)

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
	Y = torch.tensor(np.array(label_list, dtype=np.float32))
	X = torch.tensor(np.array(feat_list, dtype=np.float32))

	dataLoader = torch.utils.data.DataLoader(SelfDataLoader(X, Y), batch_size=30, shuffle=True, drop_last=False, num_workers=2)
	lr = args.lr
	epochs = args.epochs

	# pass array for svm training
	# one-versus-rest multiclass strategy
	model = NNModel(args.feat_dim)
	totalBatch = len(dataLoader)
	NUM_BATCH_WARM_UP = totalBatch * 5
	criterion = FocalLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  epochs * totalBatch)
	tqdmDesc = tqdm(total=epochs)
	for epoch in range(epochs) :
		model.train()
		validBatchId = random.sample(range(0, len(dataLoader)), len(dataLoader) // 10)
		for batch_idx, (data, target) in enumerate(dataLoader):
			optimizer.zero_grad()
			if batch_idx in validBatchId :
				continue
			outputs = model(data)
			loss = criterion(outputs, target)
			loss.backward()
			optimizer.step()
			scheduler.step()
		model.eval()
		accCnt = 0
		for i in range(X.shape[0]) :
			modelY = model(torch.tensor(X[i])).clone().detach().requires_grad_(True)
			if torch.argmax(modelY) == Y[i] : accCnt += 1
		tqdmDesc.set_postfix(acc='{:.6f}'.format(accCnt / X.shape[0]))
		tqdmDesc.update(1)
		logger.add_scalar(tag="test accuracy", scalar_value=accCnt / Y.shape[0] , global_step=epoch + 1)
		logger.flush()
	
	accCnt = 0
	for i in range(X.shape[0]) :
		modelY = model(torch.tensor(X[i])).clone().detach()
		if torch.argmax(modelY) == Y[i] : accCnt += 1
	print(f"train set accuracy: {accCnt / Y.shape[0]}")
	
	
	# save trained SVM in output_file
	pickle.dump(model, open(args.output_file, 'wb'))
	print('One-versus-rest multi-class SVM trained successfully')

