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
	
class CNN(nn.Module) :
	def __init__(self, num_classes=6) :
		super(CNN, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 3, (1, 3), (1, 0)),
			nn.ReLU(),
			nn.BatchNorm2d(32),
			# nn.LayerNorm([32, 128, 3]),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU(),
			# nn.LayerNorm([64, 128, 3]),
			nn.BatchNorm2d(64),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.AdaptiveAvgPool2d((64, 3))
		)
		self.fc = nn.Sequential(
			nn.Linear(128 * 64 * 3, 128),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
			nn.Linear(128, num_classes)
		)
		
	def forward(self, x) :
		x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
		x = self.conv(x)
		x = x.view(x.shape[0], -1)
		x = self.fc(x)
		return x
	
			