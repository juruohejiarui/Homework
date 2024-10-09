import torch.utils
import torch.utils.data
import torch.utils.data.dataset as dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import sklearn.preprocessing as preprocess

class MFCCDataset(torch.utils.data.Dataset):
	def __init__(self, X, orgX, Xfreq, Y):
		self.X = X
		self.Xlen = [len(orgX[i]) for i in range(len(X))]
		self.Xfreq = Xfreq
		self.Y = Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		labelMap = torch.zeros(10)
		labelMap[self.Y[idx]] = 1
		return (torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Xlen[idx])), labelMap
	
# 对数据进行填充，使得每个序列长度一致
def pad_sequences(sequences, maxlen=None):
	lengths = [len(seq) for seq in sequences]
	max_len = maxlen or max(lengths)
	
	padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
	for i, seq in enumerate(sequences):
		padded_sequences[i, :len(seq)] = seq
	return padded_sequences, lengths

class RNNModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes):
		super(RNNModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

		for m in self.modules():
			if isinstance(m, torch.nn.Linear) :
				torch.nn.init.orthogonal_(m.weight)


	def forward(self, x, lengths):
		# 打包序列以处理不同长度
		h0, c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).cuda(), torch.zeros((self.num_layers, x.shape[0], self.hidden_size)).cuda()
		 
		out, _ = self.lstm(x, (h0, c0))
		# print(out.shape)
		# 仅使用最后一个时间步的隐藏状态
		out = self.fc(out[:, -1, :])
		
		return out

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


# 训练模型
def train_rnn_model(X, Xfreq, Y, logger : SummaryWriter, input_size, hidden_size, num_layers, num_classes, num_epochs, batch_size, learning_rate, gradient_accumulations,
					lr_decay_interval,
					lr_decay_rate):
	# 数据预处理
	p = [i for i in range(len(X))]
	random.shuffle(p)
	newX, newXfreq, newY = [X[p[i]] for i in range(len(X))], [Xfreq[p[i]] for i in range(len(X))], [Y[p[i]] for i in range(len(X))]
	X, Xfreq, Y = newX, newXfreq, newY
	X_padded, lengths = pad_sequences(X)
	validSize = len(X) // 10
	trainSize = len(X) - validSize
	trainset = MFCCDataset(X_padded[0 : trainSize], X[0 : trainSize], Xfreq[0 : trainSize], Y[0 : trainSize])
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
	validset = MFCCDataset(X_padded[trainSize : ], X[trainSize : ], Xfreq[trainSize : ], Y[trainSize : ])
	validLoader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=False)

	def update_lr(optimizer : torch.optim.AdamW, lr) :
		for param_group in optimizer.param_groups :
			param_group['lr'] = lr
	
	# 初始化模型、损失函数和优化器
	model = RNNModel(input_size, hidden_size, num_layers, num_classes).cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
	optimizer.zero_grad()
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  num_epochs * len(trainLoader))
	curLr = learning_rate

	for epoch in tqdm(range(num_epochs)):
		lossSum = 0
		model.train()
		for i, ((inputs, inputLengths), labels) in enumerate(trainLoader):
			# 前向传播
			outputs = model(inputs.cuda(), inputLengths)
			loss = criterion(outputs, labels.cuda())
			lossSum += loss
			# 反向传播和优化
			loss.backward()
			
			if (i + 1) % gradient_accumulations == 0 :
				optimizer.step()
				optimizer.zero_grad()

			# if (i+1) % 10 == 0:
			#	 print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
		if (epoch + 1) % lr_decay_interval == 0 :
			curLr *= lr_decay_rate
			update_lr(optimizer, curLr)
			logger.add_scalar("loss/rnn", lossSum.item(), epoch + 1)
			logger.add_scalar("accuracy/validate", test_rnn_model(model, validLoader), epoch + 1)
			logger.add_scalar("accuracy/train", test_rnn_model(model, trainLoader), epoch + 1)
		if (epoch + 1) % 100 == 0 : torch.save(model, f"backups/model-epoch{epoch + 1}")
		

	print("Training finished.")
	return model

def test_rnn_model(model, validLoader):
	model.eval()  # 设置模型为评估模式，禁用dropout等
	
	correct = 0
	total = 0

	with torch.no_grad():  # 测试阶段不需要计算梯度
		for (inputs, inputLengths), labels in validLoader:
			outputs = model(inputs.cuda(), inputLengths)
			predicted = torch.argmax(outputs, dim=1)  # 获取预测结果
			total += labels.shape[0]
			correct += (predicted.cpu() == labels.argmax(dim=1)).sum().item()

	accuracy = 100 * correct / total
	# print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
	return accuracy
