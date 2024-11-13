import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VideoResNet(nn.Module):
	def __init__(self, num_classes):
		super(VideoResNet, self).__init__()
		# Load pre-trained ResNet-3D models
		self.r3d_18 = models.video.r3d_18(pretrained=True)
		
		# Freeze all parameters except the last fully connected layer
		for name, param in self.r3d_18.named_parameters():
			if 'fc' not in name:
				param.requires_grad = False
		
		# Replace the last fully connected layer to accommodate the new classification task
		self.r3d_18.fc = nn.Linear(self.r3d_18.fc.in_features, num_classes)
	
	def forward(self, x):
		return self.r3d_18(x)
	
class VideoResnetLSTM(nn.Module) :
	def __init__(self, num_classes, hiddenSize=256, numLayers=3) :
		super(VideoResnetLSTM, self).__init__()
		self.r101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
		for name, param in self.r101.named_parameters() :
			if 'fc' not in name :
				param.requires_grad = False
		self.r101.fc = nn.Linear(self.r101.fc.in_features, 256)
		self.lstm = nn.LSTM(input_size=256, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
		self.fc = nn.Linear(hiddenSize, num_classes)
		self.hiddenSize = hiddenSize
	def forward(self, x : torch.Tensor) :
		x = x.permute(0, 2, 1, 3, 4)
		batchSize, seqLen, C, H, W = x.shape
		features = torch.zeros((batchSize, seqLen, 256)).cuda()
		for t in range(seqLen) :
			features[:, t, :] = self.r101(x[:, t, :, :, :])
		lstmOut, (_, _) = self.lstm(features)
		lstmOut = lstmOut[:, -1, :]

		return self.fc(lstmOut)
		
class Fusion(nn.Module) :
	def __init__(self, num_classes) :
		super(Fusion, self).__init__()
		self.r3d = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
		self.r3d.fc = nn.Linear(self.r3d.fc.in_features, 128)
		
		conv_layers = []
		self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		conv_layers += [self.conv1, self.relu1, self.bn1]

		self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(16)
		conv_layers += [self.conv2, self.relu2, self.bn2]

		self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu3 = nn.ReLU()
		self.bn3 = nn.BatchNorm2d(32)
		conv_layers += [self.conv3, self.relu3, self.bn3]

		self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(64)
		conv_layers += [self.conv4, self.relu4, self.bn4]

		self.mfcc_ap = nn.AdaptiveAvgPool2d(output_size=1)
		self.mfcc_classification = nn.Linear(in_features=64, out_features=128)

		self.mfcc_conv = nn.Sequential(*conv_layers)

		self.comb = nn.Linear(256, num_classes)

	def forward(self, frames, mfcc) :
		frames = self.r3d(frames)
		mfcc = mfcc[:, torch.newaxis, :, :]
		mfcc = self.mfcc_conv(mfcc)
		mfcc = self.mfcc_ap(mfcc)
		mfcc = self.mfcc_classification(mfcc.view(mfcc.shape[0], -1))

		frames = F.leaky_relu(frames)
		mfcc = F.leaky_relu(mfcc)
		return self.comb(torch.stack([frames, mfcc], dim=1))
		

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