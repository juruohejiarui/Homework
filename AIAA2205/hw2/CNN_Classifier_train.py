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
	def __init__(self, root, df : list[tuple[str, int]], transform=None):
		self.root = root
		self.transforms = transform
		self.df = df
		self.imgList : list[list[str]] = [None for _ in range(len(self.df))]
		self.imgPath : list[str] = [None for _ in range(len(self.df))]
		self.labels = torch.zeros(len(self.df))
		self.labelList = []
		self.mxSeqLength = 0
		for i in tqdm(range(len(self.df))) :
			vid, label = self.df[i]
			self.imgPath[i] = os.path.join(self.root, f"{vid}")
			self.imgList[i] = sorted(os.listdir(self.imgPath[i]))
			self.mxSeqLength = max(self.mxSeqLength, len(self.imgList[i]))
			self.labels[i] = label
			self.labelList.append(label)
		self.labelList = torch.tensor(self.labelList)
		self.classes = sorted(self.labelList.unique())

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		img_list = self.imgList[index]
		imgs = []

		for img in img_list :
			img_path = os.path.join(self.imgPath[index], img)
 
			img = Image.open(img_path).convert('RGB')
			if self.transforms is not None:
				img = self.transforms(img)
			imgs.append(img)

		imgs = torch.stack(imgs)
		pad_imgs = torch.zeros((self.mxSeqLength, 3, 224, 224))
		pad_imgs[0 : imgs.size(0)] = imgs
		return pad_imgs, self.classes.index(self.labels[index])

def train_model(
				logger : SummaryWriter, 
				model_name : str,
				train_loader : DataLoader, val_loader : DataLoader, 
				epochs : int, lr : float, momentum : float, weight_decay : float,
				convDesc : list[tuple[int, int, int, int]], ConvOutputSize : int,
				hiddenSize : int, numLayers : int,
				numClass : int) :
	model = models.CNN_LSTMModel(
		convDesc=convDesc,
		convOutputSize=ConvOutputSize,
		hiddenSize=hiddenSize,
		numLayers=numLayers,
		bidirectional=False,
		numClass=numClass
	).cuda()
	
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	criterion = FocalLoss()
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader), 1e-6)

	for epoch in tqdm(range(epochs)) :
		tot, acc, lossSum = 0, 0, 0
		model.train()
		for (img, target) in train_loader :
			optimizer.zero_grad()
			img, target = img.cuda(), target.cuda()
			output : torch.Tensor = model(img)
			loss : torch.Tensor = criterion(output, target)
			lossSum += loss
			loss.backward()
			
			pred : torch.Tensor = output.argmax(1)
			tot += pred.size(0)
			acc += (pred == target).sum().item()
			
			optimizer.step()
			scheduler.step()
		
		logger.add_scalar("accuracy/train", acc * 100 / tot, epoch + 1)
		logger.add_scalar("loss/train", lossSum, epoch + 1)
		model.eval()

		tot, acc, lossSum = 0, 0, 0
		model.eval()
		for (img, target) in val_loader :
			img, target = img.cuda(), target.cuda()
			output : torch.Tensor = model(img)
			loss : torch.Tensor = criterion(output, target)
			lossSum += loss
			
			pred : torch.Tensor = output.argmax(1)
			tot += pred.size(0)
			acc += (pred == target).sum().item()
		
		logger.add_scalar("accuracy/valid", acc * 100 / tot, epoch + 1)
		logger.add_scalar("loss/valid", lossSum, epoch + 1)

		if (epoch + 1) % 100 == 0 :
			torch.save(model, f"backups/{model_name}{epoch + 1}")
	torch.save(model, f"models/{model_name}")

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--batchSize", default=16, type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("loggerSuffix", type=str)


if __name__ == "__main__" :
	args = parser.parse_args()
	model_name = args.model_name
	lr = args.lr
	epochs = args.epochs
	batchSize = args.batchSize
	momentum = args.momentum
	weight_decay = args.weight_decay
	
	logger = SummaryWriter(f"run/{model_name}-{args.loggerSuffix}", flush_secs=1)

	# You can add data augmentation here
	transform = transforms.Compose([
				transforms.Resize((224, 224)),
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
	df_test = [df.iloc[p[index], :] for index in range(trainSize, len(df))]
	train_data = MyDataset("./data/video_frames_30fpv_320p", df_train, transform)
	val_data = MyDataset("./data/video_frames_30fpv_320p", df_test, transform)

	print(f"train size : {len(train_data)} valid size : {len(val_data)}")

	train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True, num_workers=3)
	val_loader = DataLoader(val_data, batch_size=batchSize, shuffle=False, num_workers=3)

	train_model(logger=logger,
			 model_name=model_name,
			 train_loader=train_loader,
			 val_loader=val_loader,
			 lr=lr,
			 epochs=epochs,
			 momentum=momentum,
			 weight_decay=weight_decay,
			 convDesc=[(8,5,4,2),(16,3,2,2),(32,2,1,1),(64,2,1,1)],
			 ConvOutputSize=1,
			 hiddenSize=64,
			 numLayers=3,
			 numClass=10)

