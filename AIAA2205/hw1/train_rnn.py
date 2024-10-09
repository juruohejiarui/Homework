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
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.preprocessing as preprocessing

# Train SVM

import torch.nn as nn
import torch.nn.functional as F

import rnn

parser = argparse.ArgumentParser()
parser.add_argument("mfcc_dir")
parser.add_argument("mfcc_dim", type=int)
parser.add_argument("freq_dir")
parser.add_argument("freq_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--mfcc_appendix", default=".mfcc.csv")
parser.add_argument("--freq_appendix", default=".csv")
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--hidden_size", default=240, type=int)
parser.add_argument("--num_layers", default=3, type=int)
parser.add_argument("--batch_size", default=30, type=int)
parser.add_argument("--gradient_accumulations", default=16, type=int)
parser.add_argument("--lr_decay_interval", default=5, type=int)
parser.add_argument("--lr_decay_rate", default=0.8, type=int)
parser.add_argument("log_prefix")


if __name__ == '__main__':
	args = parser.parse_args()
	lr = args.lr
	epochs = args.epochs
	mfcc_dim = args.mfcc_dim
	hidden_size = args.hidden_size
	num_layers = args.num_layers
	batch_size = args.batch_size
	gradient_accumulations = args.gradient_accumulations
	lr_decay_interval = args.lr_decay_interval
	lr_decay_rate = args.lr_decay_rate

	scaler : preprocessing.StandardScaler = pickle.load(open('models/scaler', 'rb'))

	logger = SummaryWriter(comment=args.log_prefix, filename_suffix=".tfevent", flush_secs=1)

	# 1. read all features in one array.
	fread = open(args.list_videos, "r")
	seqList, freqList = [], []
	# labels are [0-9]
	label_list = []
	# load video names and events in dict
	df_videos_label = {}
	for line in open(args.list_videos).readlines()[1:]:
		video_id, category = line.strip().split(",")
		df_videos_label[video_id] = category

	for line in fread.readlines()[1:]:
		video_id = line.strip().split(",")[0]
		seq_filepath = os.path.join(args.mfcc_dir, video_id + args.mfcc_appendix)
		freq_filepath = os.path.join(args.freq_dir, video_id + args.freq_appendix)
		# for videos with no audio, ignore
		if os.path.exists(seq_filepath):
			seqList.append(np.genfromtxt(seq_filepath, delimiter=";", dtype="float"))
			freqList.append(np.genfromtxt(freq_filepath, delimiter=';', dtype='float'))
			label_list.append(int(df_videos_label[video_id]))
			

	print("number of samples: %s" % len(seqList))

	Y = torch.tensor(np.array(label_list, dtype=np.float32)).long()
	model = rnn.train_rnn_model(
		seqList, scaler.transform(np.array(freqList)), Y, 
		logger, 
		input_size=mfcc_dim, 
		hidden_size=hidden_size, 
		num_layers=num_layers, 
		num_classes=10, 
		num_epochs=epochs, 
		batch_size=batch_size,
		learning_rate=lr,
		gradient_accumulations=gradient_accumulations,
		lr_decay_interval=lr_decay_interval,
		lr_decay_rate=lr_decay_rate)
			
	
	# save trained SVM in output_file
	pickle.dump(model, open(args.output_file, 'wb'))
	pickle.dump(scaler, open("models/scaler", "wb"))
	print('NN trained successfully') 

