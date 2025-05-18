import torch.utils.data as data
import numpy as np
import os
import sklearn.preprocessing
import torch

class DataSet(data.Dataset) :
	def __init__(self, data_path, name : str) :
		def load_seq(seq_name) :
			seq = np.loadtxt(os.path.join(data_path, name, "Inertial Signals", f"{seq_name}_{name}.txt"), dtype=np.float32)
			return seq.reshape(seq.shape[0], seq.shape[1], 1)

		acc_x = load_seq("body_acc_x")
		acc_y = load_seq("body_acc_y")
		acc_z = load_seq("body_acc_z")
		gyro_x = load_seq("body_gyro_x")
		gyro_y = load_seq("body_gyro_y")
		gyro_z = load_seq("body_gyro_z")
		mag_x = load_seq("total_acc_x")
		mag_y = load_seq("total_acc_y")
		mag_z = load_seq("total_acc_z")


		self.input = np.concatenate((acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z), axis=2)
		labels = np.loadtxt(os.path.join(data_path, name, f"y_{name}.txt"))
		# one-hot encoding
		self.label = sklearn.preprocessing.LabelBinarizer().fit_transform(labels)
		self.label = torch.tensor(self.label, dtype=torch.float32)
	
	def __len__(self) :
		return len(self.input)
	
	def __getitem__(self, index) :
		return self.input[index], self.label[index]
	
if __name__ == "__main__" :
	data_path = "./Data"
	train_set = DataSet(data_path, "train")
	test_set = DataSet(data_path, "test")
	print(f"{train_set[0][0].shape}")