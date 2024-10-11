import torch.utils
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random

class MFCCDataset(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		labelMap = torch.zeros(10)
		labelMap[self.Y[idx]] = 1
		return (torch.tensor(self.X[idx], dtype=torch.float32)), labelMap
	
# padding the sequences and fill zero to the end
def pad_sequences(sequences, maxlen=None):
	lengths = [len(seq) for seq in sequences]
	max_len = maxlen or max(lengths)
	
	padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
	for i, seq in enumerate(sequences):
		padded_sequences[i, :len(seq)] = seq
	return padded_sequences, lengths
	
class CNNModel(nn.Module) :
	def __init__(self, num_classes):
		super().__init__()
		conv_layers = []
		self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		self.relu1 = nn.ReLU()
		# batch normalized 
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

		# "compress" the huge feature matrix into 64 features
		self.ap = nn.AdaptiveAvgPool2d(output_size=1)
		
		self.classification = nn.Linear(in_features=64, out_features=num_classes)

		self.conv = nn.Sequential(*conv_layers)
	def forward(self, x):
		x = x[: , torch.newaxis, : , :]
		x = self.conv(x)

		# flatten
		x = self.ap(x)
		x = x.view(x.shape[0], -1)

		x = self.classification(x)

		return x

# this function is copied from my UBRTP project
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


# model training
def train_cnn_model(X, Y, logger : SummaryWriter, num_classes, num_epochs, batch_size, learning_rate, momentum):
	# split the dataset into two part randomly, actually will cause some problems but I ignore them
	p = [i for i in range(len(X))]
	random.shuffle(p)
	newX, newY = [X[p[i]] for i in range(len(X))], [Y[p[i]] for i in range(len(X))]
	X, Y = newX, newY
	# no need to record the length of original sequence
	X_padded, _ = pad_sequences(X)
	validSize = len(X) // 10
	trainSize = len(X) - validSize
	trainset = MFCCDataset(X_padded[0 : trainSize], Y[0 : trainSize])
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
	validset = MFCCDataset(X_padded[trainSize : ], Y[trainSize : ])
	validLoader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=False)
	
	# initialize the model
	model = CNNModel(num_classes=num_classes).cuda()

	criterion = FocalLoss()
	# SGDM
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
	# learning rate scheduler, multiple the base learing rate with a consine function 
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  num_epochs * len(trainLoader))
	curLr = learning_rate

	for epoch in tqdm(range(num_epochs)):
		lossSum, cor, tot = 0, 0, 0
		model.train()
		for i, (inputs, labels) in enumerate(trainLoader):
			inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()
			# forwad
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			lossSum += loss
			# backward
			loss.backward()

			pred = torch.argmax(outputs, dim=1)
			cor += (pred == labels.argmax(dim=1)).sum().item()
			tot += len(labels)
			
			optimizer.step()
			scheduler.step()

		logger.add_scalar("loss/train", lossSum.item(), epoch + 1)
		logger.add_scalar("accuracy/validate", test_cnn_model(model, validLoader), epoch + 1)
		logger.add_scalar("accuracy/train", cor * 100 / tot, epoch + 1)
		
	print("Training finished.")
	return model

# this function is used to generate log
def test_cnn_model(model, validLoader):
	# set to evaluation mode
	model.eval()
	
	correct = 0
	total = 0
	
	with torch.no_grad():
		for inputs, labels in validLoader:
			outputs = model(inputs.cuda())
			predicted = torch.argmax(outputs, dim=1)
			total += labels.shape[0]
			correct += (predicted.cpu() == labels.argmax(dim=1)).sum().item()

	accuracy = 100 * correct / total
	return accuracy
