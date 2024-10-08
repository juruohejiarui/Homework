#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument('model_file')#mlp模型导入
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

	args = parser.parse_args()
	mlp = pickle.load(open(args.model_file,"rb"))
	scaler : StandardScaler = pickle.load(open('models/scaler', 'rb'))

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
		# for videos with no audio, ignored in training
		if os.path.exists(feat_filepath):
			feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

			label_list.append(int(df_videos_label[video_id]))

	print("number of samples: %s" % len(feat_list))
	y = np.array(label_list)
	X = scaler.transform(np.array(feat_list))
	

	# TA: write your code here 
	bagging_model = BaggingClassifier(
		mlp,
		n_estimators=20,
		n_jobs=-1,		
		random_state=42
	)
	param_dist = {
		'n_estimators': [5,10,20, 30],
		'max_samples': [0.5, 0.7, 1.0],
		'max_features': [0.5, 0.7, 1.0],
		'bootstrap': [True, False],
		'bootstrap_features': [True, False],
	}
	random_search = RandomizedSearchCV(
		estimator=bagging_model,
		param_distributions=param_dist,
		n_iter=10,
		scoring='accuracy',
		cv=3,
		n_jobs=-1,
		random_state=42,
		verbose=1
	)

	random_search.fit(X,y)

	best_bagging_model = random_search.best_estimator_
	
	# save trained MLP in output_file
	pickle.dump(best_bagging_model, open(args.output_file, 'wb'))
	print('Bagging model with pre-trained MLP classifier saved successfully')