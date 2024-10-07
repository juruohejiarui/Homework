import numpy as np
import os
import pickle
import argparse
import sys
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler

import scipy.stats as sci

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--models_num", type=int, default=20)

if __name__ == "__main__" :
	args = parser.parse_args()

	models_num = args.models_num

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
	
	scaler = StandardScaler()
	X = np.array(feat_list)
	X = scaler.fit_transform(X)
	Y = np.array(label_list)

	mlps = [MLPClassifier(
		hidden_layer_sizes=(2048, 100, 50),
		learning_rate_init=1e-3,
		max_iter=300, early_stopping=True, tol=1e-4) for _ in range(models_num)]

	param_dist = {
		'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (200, 100), (100, 50, 25), (100, 50, 30)],
		'activation': ['relu', 'tanh', 'logistic','identity'],
		'solver': ['adam', 'sgd', 
			#  'lbfgs'
			],
		'alpha': sci.uniform(1e-5, 1e-2),
		'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': sci.uniform(1e-4, 1e-2), 
		'max_iter': [500, 1000],
	}
	sample_weights = np.ones(len(X)) / len(X)
	bstMlps : list[MLPClassifier] = []
	
	alpha = 1
	for i in range(models_num) :
		indices = np.random.choice(len(X), size=len(X), p=sample_weights)
		subX, subY = X[indices], Y[indices]
		random_search = RandomizedSearchCV(
			estimator=mlps[i],
			param_distributions=param_dist,
			n_iter=50,	# Number of parameter settings to sample
			scoring='accuracy',	# Metric to evaluate
			cv=5,	# Number of cross-validation folds
			n_jobs=-1,	# Use all available CPU cores
			# random_state=42,	# For reproducibility
			verbose=1	# Print progress messages
		)
		mlps[i].fit(subX, subY)
		bstMlps.append(mlps[i])
		model = bstMlps[-1]
		Ypred = model.predict(X)
		error = np.average((Ypred != Y), weights=sample_weights)

		print(f"model{i}: error:{error}")
		
		if error > 0 and error < 1 :
			model_weight = alpha * np.log((1 - error) / error)
		else :
			model_weight = 1
		
		sample_weights *= np.exp(model_weight * (Ypred != Y))
		sample_weights /= np.sum(sample_weights)

	pickle.dump(bstMlps, open(args.output_file, 'wb'))
	pickle.dump(scaler, open('models/scaler', 'wb'))
	print('Bmlp classifier trained successfully')