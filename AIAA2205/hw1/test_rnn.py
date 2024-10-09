#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
import numpy as np
import rnn
import torch

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("mfcc_dir")
parser.add_argument("mfcc_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".mfcc.csv")

if __name__ == '__main__':

	args = parser.parse_args()

	# 1. load svm model
	model : rnn.CNNModel = pickle.load(open(args.model_file, "rb"))
	scaler : StandardScaler = pickle.load(open('models/scaler', "rb"))

	# 2. Create array containing features of ach sample
	fread = open(args.list_videos, "r")
	feat_list = []
	video_ids = []
	for line in fread.readlines():
		# HW00006228
		video_id = os.path.splitext(line.strip())[0]
		video_ids.append(video_id)
		mfcc_filepath = os.path.join(args.mfcc_dir, video_id + args.feat_appendix)
		if not os.path.exists(mfcc_filepath):
			feat_list.append(np.zeros((10, 39)))
		else:
			feat_list.append(np.genfromtxt(mfcc_filepath, delimiter=";", dtype='float'))

	# X = np.array(feat_list, dtype=np.float32)
	# X = scaler.transform(X)
	# print(X.shape)
	# 3. Get scores with trained svm model
	# (num_samples, num_class)
	model.eval()
	scoress = torch.zeros((len(feat_list), 10))
	xlen = torch.tensor([len(x) for x in feat_list])
	for i in range(len(feat_list)) :
		x = torch.tensor(feat_list[i])
		x = x.reshape((1, x.shape[0], x.shape[1]))
		scoress[i, : ] = model(x.float().cuda())

	# 4. save the argmax decisions for submission
	with open(args.output_file, "w") as f:
		f.writelines("Id,Category\n")
		for i in range(len(feat_list)) :
			predicted_class = torch.argmax(scoress[i])
			f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
