import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import models
import argparse
import random
from torchstat import stat

from tqdm import tqdm
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
	
class MyDataset(Dataset):
	def __init__(self, root, df : list[tuple[str, int]], img_seqlen : int, img_size : int, transform=None):
		self.root = root
		self.transforms = transform
		self.df = df
		self.imgList : list[list[str]] = [None for _ in range(len(self.df))]
		self.imgPath : list[str] = [None for _ in range(len(self.df))]
		self.imgSize = img_size
		self.imgSeqLen = img_seqlen
		self.labels = torch.zeros(len(self.df))
		self.mxSeqLength = 0
		for i in tqdm(range(len(self.df))) :
			vid, label = self.df[i]
			self.imgPath[i] = os.path.join(self.root, f"{vid}")
			self.imgList[i] = sorted(os.listdir(self.imgPath[i]))
			self.mxSeqLength = max(self.mxSeqLength, len(self.imgList[i]))
			self.labels[i] = label
		self.labels = self.labels.long()

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		img_list = self.imgList[index]
		imgs = []

		for img in img_list[len(img_list) // 2 - self.imgSeqLen // 2 : len(img_list) // 2 + self.imgSeqLen // 2 + self.imgSeqLen % 2]:
			img_path = os.path.join(self.imgPath[index], img)
 
			img = Image.open(img_path).convert('RGB')
			if self.transforms is not None:
				img = self.transforms(img)
			imgs.append(img)
		# while len(imgs) < self.mxSeqLength : imgs.append(torch.zeros((3, img_size, img_size)))
		labelMap = torch.zeros(10)
		labelMap[self.labels[index]] = 1

		imgs = torch.stack(imgs)
		return imgs, labelMap

def train_model(
				model : nn.Module,
				logger : SummaryWriter, 
				model_name : str,
				train_loader : DataLoader, val_loader : DataLoader, 
				adamEpochs : int = 6,
				epochs : int = 300, lr : float = 1e-3, momentum : float = 0.78, weight_decay : float = 0.1,
				) :
	model = model.cuda()
	
	adamOptimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	sgdOptimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	criterion = FocalLoss()
	sgdScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sgdOptimizer, (epochs - adamEpochs) * len(train_loader), 1e-7)

	for epoch in tqdm(range(epochs)) :
		tot, acc, lossSum = 0, 0, 0
		model.train()
		optimizer = sgdOptimizer if epoch >= adamEpochs else adamOptimizer
		scheduler = sgdScheduler if epoch >= adamEpochs else None
		for (img, target) in train_loader :
			
			optimizer.zero_grad()
			img, target = img.cuda(), target.cuda()
			output : torch.Tensor = model(img)
			loss : torch.Tensor = criterion(output, target)
			lossSum += loss
			loss.backward()
			
			pred : torch.Tensor = output.argmax(1)
			tot += pred.size(0)
			acc += (pred == target.argmax(1)).sum().item()
			
			optimizer.step()
			if scheduler != None : scheduler.step()
		
		logger.add_scalar("accuracy/train", acc * 100 / tot, epoch + 1)
		logger.add_scalar("loss/train", lossSum, epoch + 1)

		with torch.no_grad() :
			model.eval()

			tot, acc, lossSum = 0, 0, 0
			for (img, target) in val_loader :
				img, target = img.cuda(), target.cuda()
				output : torch.Tensor = model(img)
				
				pred : torch.Tensor = output.argmax(1)
				tot += pred.size(0)
				acc += (pred == target.argmax(1)).sum().item()
			
			logger.add_scalar("accuracy/valid", acc * 100 / tot, epoch + 1)

		if (epoch + 1) % 100 == 0 :
			torch.save(model, f"backups/{model_name}{epoch + 1}")
	torch.save(model, f"models/{model_name}")

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("--img_seqlen", default=1, type=int)
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("--lr", default=3e-5, type=float)
parser.add_argument("--adamEpochs", default=6, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batchSize", default=25, type=int)
parser.add_argument("--momentum", default=0.78, type=float)
parser.add_argument("--weight_decay", default=0.12, type=float)
parser.add_argument("loggerSuffix", type=str)

def get_parameter_number(model):
	total_num = sum(p.numel() for p in model.parameters())
	trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__" :
	args = parser.parse_args()
	model_name = args.model_name
	img_seqlen = args.img_seqlen
	img_size = args.img_size
	lr = args.lr
	adamEpochs = args.adamEpochs
	epochs = args.epochs
	batchSize = args.batchSize
	momentum = args.momentum
	weight_decay = args.weight_decay
	
	logger = SummaryWriter(f"run/{model_name}-{args.loggerSuffix}", flush_secs=1)

	# You can add data augmentation here
	transform = transforms.Compose([
				transforms.Resize((img_size, img_size)),
				transforms.ToTensor()
			])
	print("loading data...")

	# split manually
	df = pd.read_csv("./data/trainval.csv", header=None, skiprows=1)
	p = [i for i in range(len(df))]
	random.shuffle(p)
	
	validSize = len(df) // 10
	trainSize = len(df) - validSize
	df_train = [df.iloc[p[index], : ] for index in range(trainSize)]
	df_valid = [df.iloc[p[index], : ] for index in range(trainSize, len(df))]
	train_data = MyDataset("./data/video_frames_30fpv_320p", df_train, img_seqlen, img_size, transform)
	val_data = MyDataset("./data/video_frames_30fpv_320p", df_valid, img_seqlen, img_size, transform)

	print(f"train size : {len(train_data)} valid size : {len(val_data)}")

	train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True, num_workers=15)
	val_loader = DataLoader(val_data, batch_size=batchSize, shuffle=False, num_workers=15)

	## CNN_LSTM
	cnn3d = models.CNN3D(
		inChannel=3,
		hiddenSize=(512, 100),
		numClass=10
	)
	resnetLSTM = models.ResnetLSTM(
		resOutputSize=2048,
		numLayers=3,
		hiddenSize=256,
		numClasses=10
	)
	resnet = models.Resnet(numClass=10)
	train_model(
			model=resnet,
			logger=logger,
			model_name=model_name,
			train_loader=train_loader,
			val_loader=val_loader,
			lr=lr,
			adamEpochs=adamEpochs,
			epochs=epochs,
			momentum=momentum,
			weight_decay=weight_decay,
			)

