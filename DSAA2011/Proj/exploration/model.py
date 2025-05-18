import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
	def __init__(self, gamma=2, weight=None):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.weight = weight

	def forward(self, inputs, targets):
		ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
		pt = torch.exp(-ce_loss)  # 计算预测的概率
		focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
		return focal_loss
	
class LSTMCNN(nn.Module) :
	def __init__ (self, n_input=9, num_classes=6) :
		super(LSTMCNN, self).__init__()
		self.lstm = nn.LSTM(input_size=n_input, hidden_size=32, num_layers=2, batch_first=True)
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=2, stride=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.BatchNorm2d(256)
		)
		self.fc = nn.Sequential(
			nn.Linear(256, num_classes),
			nn.Softmax(dim=1)
		)

	def forward(self, x : torch.Tensor) :
		batch_size = x.size(0)
		x, _ = self.lstm(x)
		x = x.unsqueeze(1)

		x = self.conv1(x)

		x = x.view(batch_size, -1)
		x = self.fc(x)
		return x

	