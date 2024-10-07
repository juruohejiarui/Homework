#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import pickle
import argparse
import sys
import random
import scipy.stats as sci

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

	args = parser.parse_args()

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
			feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float64"))

			label_list.append(int(df_videos_label[video_id]))

	print("number of samples: %s" % len(feat_list))
	y = np.array(label_list)
	X = np.array(feat_list)

	scaler = StandardScaler()
	
	X = scaler.fit_transform(X)

		# TA: write your code here 
	model = MLPClassifier(max_iter=500, early_stopping=True, tol=1e-5)
	param_dist = {
		'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (200, 100), (100, 50, 25), (100, 50, 30)],
		'activation': ['relu', 'tanh', 'logistic','identity'],
		'solver': ['adam', 'sgd', 
			 'lbfgs'
			],
		'alpha': sci.uniform(1e-5, 1e-2),
		'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': sci.uniform(1e-4, 1e-2), 
		'max_iter': [1000],
	}

	random_search = RandomizedSearchCV(
		estimator=model,
		param_distributions=param_dist,
		n_iter=60,	# Number of parameter settings to sample
		scoring='accuracy',	# Metric to evaluate
		cv=5,	# Number of cross-validation folds
		n_jobs=-1,	# Use all available CPU cores
		# random_state=42,	# For reproducibility
		verbose=1	# Print progress messages
	)
	random_search.fit(X,y)
	best_model = random_search.best_estimator_
	
	# save trained MLP in output_file
	pickle.dump(best_model, open(args.output_file, 'wb'))
	pickle.dump(scaler, open('models/scaler', 'wb'))
	print('MLP classifier trained successfully')
