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
from nnUniversal import SelfDataLoader, NNModel, RNNModel
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler

# Train SVM

import torch.nn as nn
import torch.nn.functional as F

import time

parser = argparse.ArgumentParser()
parser.add_argument("mfcc_dir")
parser.add_argument("mfcc_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden_size", default=100, type=int)
parser.add_argument("--num_layers", default=1, type=int)


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
	lr = args.lr
	epochs = args.epochs
	mfcc_dim = args.mfcc_dim
	hidden_size = args.hidden_size
	num_layers = args.num_layers
	num_time_step = args.num_time_step
	

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
		seq_filepath = os.path.join(args.mfcc_path, video_id + args.feat_appendix)
		# for videos with no audio, ignore
		if os.path.exists(seq_filepath):
			feat_list.append(np.genfromtxt(seq_filepath, delimiter=";", dtype="float"))
			label_list.append(int(df_videos_label[video_id]))

	print("number of samples: %s" % len(feat_list))
	scaler = StandardScaler()

	Y = torch.tensor(np.array(label_list, dtype=np.float32))
	model = RNNModel(mfcc_dim, hidden_size, num_layers)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.MSELoss()
	hidden_prev = torch.zeros((1, 1, hidden_size))

	for epoch in tqdm(epochs) :
		for (i, seqX, y) in enumerate(zip(feat_list, Y)) :
			for pos in range(len(seqX) - num_time_step) :
				ed = pos + hidden_size
				x = torch.tensor(seqX[pos : ed]).float().view(1, num_time_step, mfcc_dim)
				y = torch.tensor(seqX[pos + 5 : ed + 5]).float().view(1, num_time_step, mfcc_dim)
			
	
	# save trained SVM in output_file
	pickle.dump(model, open(args.output_file, 'wb'))
	pickle.dump(scaler, open("models/scaler", "wb"))
	print('NN trained successfully')

