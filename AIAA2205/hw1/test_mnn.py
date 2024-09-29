#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
import numpy as np
import nnUniversal as nn
import torch

from tqdm import tqdm

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("feat1_dir")
parser.add_argument("feat1_dim", type=int)
parser.add_argument("feat2_dir")
parser.add_argument("feat2_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

	args = parser.parse_args()

	# 1. load svm model
	model : nn.NNModel = pickle.load(open(args.model_file, "rb"))

	# 2. Create array containing features of each sample
	fread = open(args.list_videos, "r")
	feat1_list, feat2_list = [], []
	video_ids = []
	for line in fread.readlines():
		# HW00006228
		video_id = os.path.splitext(line.strip())[0]
		video_ids.append(video_id)
		feat1_filepath = os.path.join(args.feat1_dir, video_id + args.feat_appendix)
		feat2_filepath = os.path.join(args.feat2_dir, video_id + args.feat_appendix)
		if not os.path.exists(feat1_filepath):
			feat1_list.append(np.zeros(args.feat1_dim))
			feat2_list.append(np.zeros(args.feat2_dim))
		else:
			feat1_list.append(np.genfromtxt(feat1_filepath, delimiter=";", dtype='float'))
			feat2_list.append(np.genfromtxt(feat2_filepath, delimiter=";", dtype='float'))


	X1 = np.array(feat1_list, dtype=np.float32)
	X2 = np.array(feat2_list, dtype=np.float32)
	print(X1.shape)
	# 3. Get scores with trained svm model
	# (num_samples, num_class)
	model.eval()
	scoress = torch.zeros((X1.shape[0], 10))
	for i in range(X1.shape[0]) :
		scoress[i, : ] = model(torch.tensor(X1[i]).cuda(), torch.tensor(X2[i]).cuda(), False)

	# 4. save the argmax decisions for submission
	with open(args.output_file, "w") as f:
		f.writelines("Id,Category\n")
		for i, scores in enumerate(scoress):
			predicted_class = torch.argmax(scores)
			f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
