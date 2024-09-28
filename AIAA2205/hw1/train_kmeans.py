#!/bin/python 
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle 
import sys
import time
import numpy as np

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
		print("mfcc_csv_file -- path to the mfcc csv file")
		print("cluster_num -- number of cluster")
		print("output_file -- path to save the k-means model")
		exit(1)

	mfcc_csv_file = sys.argv[1]; 
	output_file = sys.argv[3]
	cluster_num = int(sys.argv[2])
	
	# 1. load all mfcc features in one array
	selection = []
	files = os.listdir(mfcc_csv_file)
	for file in files :
		path = os.path.join(mfcc_csv_file, file)
		selection.append(pd.read_csv(path, sep=';', dtype='float'))
	X = np.concatenate(selection)
	# TA: perform kmeans clustering here. get a model file variable kmeans
	kmeans = KMeans(n_clusters=cluster_num, n_init=8, random_state=42, verbose=1)
	kmeans = kmeans.fit(X)
	
	# 2. Save trained model
	pickle.dump(kmeans, open(output_file, 'wb'))
	
	print("K-means trained successfully!")
