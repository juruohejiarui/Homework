import os
import pandas as pd
from PIL import Image
from dataset import MyDataset
import models
import dataset
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define aug
transforms = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) 
])

# Load test dataset
oridf = dataset.loadDf("data/test_for_student.csv")
df = [oridf.iloc[i, :] for i in range(len(oridf))]

test_dataset = dataset.MyDataset('data/hw3_32fpv', df, stage="val", transform=transforms)

print('dataset loaded')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
print(f"Length of test loader: {len(test_loader)}")

# Load model
model = models.VideoTransformer(num_classes=10).cuda()
model.load_state_dict(torch.load('models/transformer_32fpv.pth'))

# Load video ID
fread = open("data/test_for_student.label", "r")
video_ids = [os.path.splitext(line.strip())[0] for line in fread.readlines()]

# Val stage
model.eval()
result = []
with torch.no_grad():
	for data in tqdm(test_loader):
		inputs, labels = data
		inputs, labels = inputs.cuda(), labels.cuda()
		outputs = model(inputs)
		predicted = torch.argmax(outputs.data, 1)
		result.extend(predicted.cpu().numpy())

# Save result
with open('output/result_transformer.csv', "w") as f:
	f.writelines("Id,Category\n")
	for i, pred_class in enumerate(result):
		f.writelines(f"{video_ids[i]},{pred_class}\n")