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

class CNN3D(nn.Module) :
	def __init__(self,
				inChannel : int,
				hiddenSize : list[int],
				numClass : int
				) :
		super(CNN3D, self).__init__()
		self.conv = nn.Sequential()
		self.conv.append(nn.Conv3d(inChannel, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(32))
		self.conv.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

		self.conv.append(nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(64))
		self.conv.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

		self.conv.append(nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(128))
		self.conv.append(nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(128))
		self.conv.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

		self.conv.append(nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(256))
		self.conv.append(nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(256))
		self.conv.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

		self.conv.append(nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(256))
		self.conv.append(nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)))
		self.conv.append(nn.ReLU())
		self.conv.append(nn.BatchNorm3d(256))
		self.conv.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

		self.fc = nn.Sequential()
		self.convOutputSize = 16384
		for i in range(len(hiddenSize) + 1) :
			self.fc.append(nn.Linear(self.convOutputSize if i == 0 else hiddenSize[i - 1], hiddenSize[i] if i != len(hiddenSize) else numClass))
			self.fc.append(nn.LeakyReLU())
	def forward(self, x) :
		x = self.conv(x)
		# print(x.shape)
		x = self.fc(x.flatten(1))
		return x
		
class ResNetLSTM(nn.Module) :
	def __init__(self) :
		self.resnet = torch.resnet
