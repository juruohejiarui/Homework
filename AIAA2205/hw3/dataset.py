import os
import random
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

gloDf = None

def loadDf(csv_file : str) :
	return pd.read_csv(csv_file, header=None, skiprows=1)

class MyDataset(Dataset):
	def __init__(self, root, df, stage="train", transform=None):
		self.root = root
		self.transforms = transform
		self.df = df
		self.stage = stage
		self.files = [self.df[i][0] for i in range(len(self.df))]
		self.frame_file_list = []
		self.labels = []

		for i in range(len(self.files)) :
			vid = self.files[i].split(".mp4")[0]
			img_list = os.listdir(os.path.join(self.root, f"{vid}.mp4"))
			img_list = sorted(img_list)
			label = torch.zeros(10)
			label[self.df[i][1]] = 1
			self.frame_file_list.append(img_list)
			self.labels.append(label)
			if i == 0 : print(self.files[0], self.frame_file_list[0], self.labels[0])

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		vid = self.files[index]
		img_16fpv = [self.transforms(Image.open(os.path.join(self.root, f"{vid}.mp4", img_path)).convert('RGB')) for img_path in self.frame_file_list[index]]
		img_16fpv_tensor = torch.stack(img_16fpv).permute(1,0,2,3)
		return img_16fpv_tensor, self.labels[index]