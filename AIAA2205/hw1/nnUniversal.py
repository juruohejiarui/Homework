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
		return len(self.data)

class CNNDataLoader(dataset.Dataset) :
	def __init__(self, data, dataIdx, dataLabel, fileList, row, col) :
		self.data : torch.Tensor = data
		self.dataIdx : torch.Tensor = dataIdx
		self.label : torch.Tensor = dataLabel
		self.row : int = row
		self.col : int = col
		self.files = fileList
		
	
	def __getitem__(self, index) :
		X2 = torch.zeros((1, self.row, self.col))
		rawData = self.files[self.dataIdx[index].long()]
		# print(rawData.shape)
		X2[0, 0 : rawData.shape[0], 0 : rawData.shape[1]] = rawData
		X1 = self.data[index]
		labelMap = torch.zeros(10)
		labelMap[self.label[index].long()] = 1
		return (X1, X2), labelMap
	def __len__(self) :
		return len(self.data)

class MNNDataLoader(dataset.Dataset) :
	def __init__(self, data1, data2, dataLabel) :
		super(MNNDataLoader, self).__init__()
		self.data1 : torch.Tensor = data1
		self.data2 : torch.Tensor = data2
		self.label : torch.Tensor = dataLabel
		
	
	def __getitem__(self, index) :
		X1, X2 = self.data1[index], self.data2[index]
		labelMap = torch.zeros(10)
		labelMap[self.label[index].long()] = 1
		return (X1, X2), labelMap
	def __len__(self) :
		return len(self.data1)
	
class NNModel(nn.Module) :
	def __init__(self, input_dim) :
		super(NNModel, self).__init__()
		self.fc1 = nn.Linear(input_dim, 12800)
		self.fc2 = nn.Linear(12800, 1024)
		self.fc3 = nn.Linear(1024, 256)
		self.fc4 = nn.Linear(256, 10)
	def forward(self, X) :
		X = F.leaky_relu(self.fc1(X))
		X = F.dropout(X)
		X = F.leaky_relu(self.fc2(X))
		X = F.leaky_relu(self.fc3(X))
		X = F.leaky_relu(self.fc4(X))
		return X

class CNNModel(nn.Module) :
	def __init__(self, channel, row, col, inputDim) :
		super(CNNModel, self).__init__()
		self.conv1 = nn.Conv2d(1, channel, (1, 40), (1, 40))
		self.fc1 = nn.Linear(channel * (row // 1) * (col // 40) + inputDim, 12800)
		self.fc2 = nn.Linear(12800, 256)
		self.fc3 = nn.Linear(256, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, X1, X2, isBatch) :
		dim = 1 if isBatch else 0
		X2 : torch.Tensor = self.conv1(X2)
		X : torch.Tensor = torch.cat((X1, torch.flatten(X2, start_dim = dim)), dim=dim)
		X = F.leaky_relu(self.fc1(X))
		X = F.dropout(X, p = 0.5)
		X = F.prelu(self.fc2(X))
		X = F.leaky_relu(self.fc3(X))
		X = F.leaky_relu(self.fc4(X))
		return X
	
class MNNModel(nn.Module) :
	def __init__(self, inputDim1, inputDim2) :
		super(MNNModel, self).__init__()
		self.fc1 = nn.Linear(inputDim1 + inputDim2, 12800)
		self.fc2 = nn.Linear(inputDim1 + inputDim2, 30)
		self.fc3 = nn.Linear(12830, 20)
		self.fc4 = nn.Linear(20, 10)

		for name, param in self.named_parameters():
			if 'weight' in name:
				nn.init.normal_(param, mean=0, std=0.02)

	def forward(self, X1, X2, isBatch) :
		dim = 1 if isBatch else 0
		X = torch.cat((X1, X2), dim=dim)
		X1 = F.mish(self.fc1(X))
		X2 = F.mish(self.fc2(X))
		X = torch.cat((X1, X2), dim=dim)
		X = F.dropout(X, p=0.1)
		X = F.leaky_relu(self.fc3(X))
		X = F.leaky_relu(self.fc4(X))
		return X