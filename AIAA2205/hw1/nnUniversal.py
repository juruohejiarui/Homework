import torch.utils
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np

class SelfDataLoader(dataset.Dataset) :
	def __init__(self, data_root, data_label):
		self.data : torch.Tensor = data_root
		self.label : torch.Tensor = data_label

	def __getitem__(self, index):
		data = self.data[index]
		labels = self.label[index]
		labelMap = torch.zeros(10)
		labelMap[labels.long()] = 1
		return data, labelMap
	
	def __len__(self):
		return self.data.shape[0]
	
class NNModel(nn.Module) :
	def __init__(self, input_dim) :
		super(NNModel, self).__init__()
		self.fc1 = nn.Linear(input_dim, 12800)
		self.drop1 = nn.Dropout(p=0.2)
		self.fc2 = nn.Linear(12800, 4096)
		self.drop2 = nn.Dropout(p=0.1)
		self.fc3 = nn.Linear(4096, 512)
		self.fc4 = nn.Linear(512, 512)
		self.fc5 = nn.Linear(512, 10)
	def forward(self, X) :
		X = F.leaky_relu(self.fc1(X))
		# X = self.drop1(X)
		X = F.leaky_relu(self.fc2(X))
		# X = self.drop2(X)
		X = F.leaky_relu(self.fc3(X))
		X = F.leaky_relu(self.fc4(X)) + X
		X = F.leaky_relu(self.fc5(X))
		return X