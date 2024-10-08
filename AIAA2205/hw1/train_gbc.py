#!/bin/python

import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import argparse
import sys
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

# Train Gradient Boosting classifier with labels

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
            feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
            label_list.append(int(df_videos_label[video_id]))

    print("number of samples: %s" % len(feat_list))
    y = np.array(label_list)
    X = np.array(feat_list)

    param_dist = {
        'n_estimators': randint(50, 100),  # Number of boosting stages
        'learning_rate': uniform(0.01, 0.2),  # Step size shrinkage
        'max_depth': randint(3, 10),  # Maximum tree depth
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 10),  # Minimum number of samples required to be at a leaf node
        'subsample': uniform(0.5, 1.0)  # Fraction of samples to be used for fitting the individual base learners
    }

    # Create Gradient Boosting model
    model = GradientBoostingClassifier()

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings to sample
        scoring='accuracy',  # Metric to evaluate
        cv=3,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,  # For reproducibility
        verbose=1  # Print progress messages
    )
    
    random_search.fit(X, y)
    best_model = random_search.best_estimator_

    # Save trained model in output_file
    pickle.dump(best_model, open(args.output_file, 'wb'))
    print('Gradient Boosting classifier trained successfully')
    print('Best params:', random_search.best_params_)