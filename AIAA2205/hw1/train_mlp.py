import numpy as np
import os
import pickle
import argparse
import sys
import sklearn.cross_decomposition
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
import sklearn.model_selection
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import sklearn

import scipy.stats as sci
import sklearn.tests

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--models_num", type=int, default=1)

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

	mlps = [(f"mlp-{i}", MLPClassifier(
		hidden_layer_sizes=(2048, 100, 50),
		learning_rate_init=1e-3,
		
		max_iter=300, early_stopping=True, tol=1e-4)) for i in range(models_num)]
	mlp = MLPClassifier(max_iter=500)

	param_dist = {
		'hidden_layer_sizes': [(800,50,25), (1000, 50, 25)],
		'activation': ['relu', 'tanh', 'logistic','identity', 'leaky_relu'],
		'solver': ['adam', 'sgd', 
			#  'lbfgs'
			],
		'alpha': sci.uniform(1e-5, 4e-3),
		'learning_rate': ['constant', 'invscaling', 'adaptive'],
		'learning_rate_init': sci.uniform(1e-3 - 1e-5, 1e-3 + 2e-5), 
		'max_iter': [1000],
	}
	random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=100, n_jobs=32 - 1)
	random_search.fit(X, Y)
	mlp = random_search.best_estimator_
	acc = sklearn.model_selection.cross_val_score(mlp, X, Y, cv=5)
	
	pickle.dump(mlp, open(args.output_file, 'wb'))
	pickle.dump(scaler, open('models/scaler', 'wb'))
	print('Bmlp classifier trained successfully, acc={0}' % (acc))
	print(f'best choice: {random_search.best_params_}')