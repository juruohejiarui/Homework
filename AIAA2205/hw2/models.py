# coding=utf-8
"""Models."""

import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self,dropout_prob=0.5):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(256 * 56 * 56, 64)
		self.fc2 = nn.Linear(64, 10)
		self.dropout_prob = dropout_prob

	def forward(self, x):
		x = self.bn1(self.pool(F.relu(self.conv1(x))))
		x = self.bn2(self.pool(F.relu(self.conv2(x))))
		x = x.view(-1, 256 * 56 * 56)
		x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_prob)
		x = self.fc2(x)
		return x

class CNN_LSTMModel(nn.Module) :
	def __init__(self, 
				 convDesc : list[tuple[int, int, int, int]], 
				 convOutputSize : int,
				 hiddenSize : int, numLayers : int, bidirectional : bool,
				 numClass : int) :
		super(CNN_LSTMModel, self).__init__()
		self.convDesc = convDesc
		self.convOutputSize = convOutputSize
		self.numClass = numClass

		seq = []
		lstChannel = 3
		for partDesc in convDesc :
			convLyr = nn.Conv2d(lstChannel, partDesc[0], kernel_size=partDesc[1], padding=partDesc[2], stride=partDesc[3])
			reluLyr = nn.ReLU()
			normLyr = nn.BatchNorm2d(partDesc[0])
			lstChannel = partDesc[0]
			seq += [convLyr, reluLyr, normLyr]
		self.conv = nn.Sequential(*seq)
		self.ap = nn.AdaptiveAvgPool2d(convOutputSize)

		self.lstm = nn.LSTM(input_size=(convOutputSize ** 2) * convDesc[-1][0], 
							hidden_size=hiddenSize, num_layers=numLayers, bidirectional=bidirectional)
		self.fc = nn.Linear(hiddenSize if bidirectional == False else hiddenSize * 2, numClass)

		self.forward = self.forward_cnn
	
	def forward_lstm(self, x : torch.Tensor) :
		pass

	def forward_cnn(self, x : torch.Tensor) :
		batchSize, seqLength, channel, sizeX, sizeY = x.shape
		x = self.conv(x.view((batchSize * seqLength, channel, sizeX, sizeY)))
		x = self.ap(x)
		x = x.view((batchSize, seqLength, -1))
		x, _ = self.lstm(x)
		x = x[:, -1, :]
		x = self.fc(x)
		return F.leaky_relu(x)
	
