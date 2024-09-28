#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster import KMeans
import sys
import pandas as pd
import time
import collections
import csv
import argparse
from tqdm import tqdm
# Generate MFCC-Bag-of-Word features for videos
# each video is represented by a single vector

parser = argparse.ArgumentParser()
parser.add_argument("kmeans_model")
parser.add_argument("cluster_num", type=int)
parser.add_argument("file_list")
parser.add_argument("--mfcc_path", default="mfcc")
parser.add_argument("--output_path", default="bof")

if __name__ == '__main__':
	args = parser.parse_args()

	# 1. load the kmeans model
	kmeans : KMeans = pickle.load(open(args.kmeans_model, "rb")) 

	# 2. iterate over each video and
	start = time.time()
	fread = open(args.file_list, "r")
 
	total_frame = args.cluster_num

	for line in tqdm(fread.readlines()):
		mfcc_path = os.path.join(args.mfcc_path, line.strip() + ".mfcc.csv")
		bof_path = os.path.join(args.output_path, line.strip() + ".csv")

		if not os.path.exists(mfcc_path):
			continue
		# (num_frames, d)
		array = numpy.genfromtxt(mfcc_path, delimiter=";")

		#  ************************************TA: Your code starts here

		## use kmeans.predict(mfcc_features_of_video)
		# (num_frames,), each row is an integer for the closest cluster center

		# create dict containing frequencies of each "code word"
	
		# {0: count_for_0, 1: count_for_1, ...}

		# normalize the frequency by dividing with frame number
		y = kmeans.predict(array)
		freq_dict = collections.Counter(y)
		list_freq = []
		tot = 0
		for i in range(total_frame):
			tot += freq_dict.get(i, 0)
		for i in range(total_frame) :
			cnt = freq_dict.get(i, 0) / tot
			list_freq.append(cnt)

		#  ************************************TA: Your code ends here

		numpy.savetxt(bof_path, numpy.array(list_freq))

	end = time.time()
	print("K-means features generated successfully!")
	print("Time for computation: ", (end - start))
