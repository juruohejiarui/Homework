import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfDataLoader(dataset.Dataset) :
	def __init__(self, data_root, data_label):
		self.data = data_root
		self.label = data_label

	def __getitem__(self, index):
		data = self.data[index]
		labels = self.label[index]
		labelMap = torch.zeros(10)
		labelMap[labels.long()] = 1
		return data, labelMap
	
	def __len__(self):
		return len(self.data)
	
class NNModel(nn.Module) :
	def __init__(self, input_dim) :
		super(NNModel, self).__init__()
		self.fc1 = nn.Linear(input_dim, 12800)
		self.fc2 = nn.Linear(12800, 256)
		self.fc3 = nn.Linear(256, 128)
		self.fc4 = nn.Linear(128, 10)
	def forward(self, X) :
		X = F.leaky_relu(self.fc1(X))
		X = F.dropout(X)
		X = F.leaky_relu(self.fc2(X))
		X = F.leaky_relu(self.fc3(X))
		X = F.leaky_relu(self.fc4(X))
		return X