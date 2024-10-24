import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from models import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm
import pickle
import argparse
import models

class MyDataset(Dataset):
	def __init__(self, root, df : list[tuple[str, int]], img_size : int, transform=None):
		self.root = root
		self.transforms = transform
		self.df = df
		self.imgList : list[list[str]] = [None for _ in range(len(self.df))]
		self.imgPath : list[str] = [None for _ in range(len(self.df))]
		self.imgSize = img_size
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

		for img in img_list[len(img_list) // 2 : len(img_list) // 2 + 1]:
			img_path = os.path.join(self.imgPath[index], img)
 
			img = Image.open(img_path).convert('RGB')
			if self.transforms is not None:
				img = self.transforms(img)
			imgs.append(img)
		labelMap = torch.zeros(10)
		labelMap[self.labels[index]] = 1

		imgs = torch.stack(imgs)
		return imgs, labelMap
	
parser = argparse.ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("--img_size", default=224, type=int)
parser.add_argument("output_path", type=str)

if __name__ == "__main__" :
	args = parser.parse_args()
	model_name = args.model_name
	img_size = args.img_size
	output_path = args.output_path
	transform = transforms.Compose([
				transforms.Resize((img_size, img_size)),
				transforms.ToTensor()
			])
	df = pd.read_csv("./data/test_for_student.csv", header=None, skiprows=1)
	dfData = [df.iloc[i, :] for i in range(len(df))]
	test_dataset = MyDataset("./data/video_frames_30fpv_320p", dfData, img_size, transform)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=15)

	# Load Model
	# net = pickle.load(open(model_name, "rb"))
	net : models.ResnetLSTM = torch.load(model_name, weights_only=False)

	# Evaluation
	net.eval()
	net.cuda()
	video_ids = [test_dataset.df[i][0] for i in range(len(test_dataset))]
	result = []
	with torch.no_grad():
		# TODO: Evaluation result here ...
		for (imgs, _) in tqdm(test_loader) :
			lblMap = net(imgs.cuda())
			lblMap = lblMap.argmax(dim=1)
			for i in range(lblMap.shape[0]) :
				result.append(lblMap[i])


	with open(output_path, "w") as f:
		f.writelines("Id,Category\n")
		for i, pred_class in enumerate(result):
			f.writelines("%s,%d\n" % (video_ids[i], pred_class))