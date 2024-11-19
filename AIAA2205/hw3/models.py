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
		self.r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		for name, param in self.r50.named_parameters() :
			if 'fc' not in name :
				param.requires_grad = False
		self.r50.fc = nn.Linear(self.r50.fc.in_features, 256)
		self.lstm = nn.LSTM(input_size=256, hidden_size=hiddenSize, num_layers=numLayers, batch_first=True)
		self.fc = nn.Linear(hiddenSize, num_classes)
		self.hiddenSize = hiddenSize
	def forward(self, x : torch.Tensor) :
		x = x.permute(0, 2, 1, 3, 4)
		batchSize, seqLen, C, H, W = x.shape
		features = torch.zeros((batchSize, seqLen, 256)).cuda()
		for t in range(seqLen) :
			features[:, t, :] = self.r50(x[:, t, :, :, :])
		lstmOut, (_, _) = self.lstm(features)
		lstmOut = lstmOut[:, -1, :]

		return self.fc(lstmOut)
class VideoTransformer(nn.Module) :
	def __init__(self, num_classes=10) :
		super(VideoTransformer, self).__init__()
		self.swin = models.video.swin_transformer.swin3d_b(weights=models.video.swin_transformer.Swin3D_B_Weights.DEFAULT)
		in_feature = self.swin.head.in_features
		for name, param in self.swin.named_parameters() :
			if not name.startswith("head") :
				param.requires_grad = False
		self.swin.head = nn.Linear(in_feature, num_classes)
	def forward(self, x) :
		return self.swin(x)

class VGGLSTM(nn.Module) :
	def __init__(self, num_classes, input_size=1000, hidden_size=400, num_layers=4) :
		super(VGGLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.input_size = input_size
		self.cnn = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.cuda()
		for param in self.cnn.parameters() :
			param.requires_grad = False
		self.fc1 = nn.Linear(512 * 7 * 7, input_size)
		self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.fc2 = nn.Linear(hidden_size, num_classes)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x : torch.Tensor) :
		x = x.permute(0, 2, 1, 3, 4)
		batch_size, seq_len, C, H, W = x.size()
		frame_features = torch.zeros((batch_size, seq_len, self.input_size)).cuda()
		for t in range(seq_len) :
			feature = self.cnn(x[:, t, :, :, :])
			feature = feature.view(batch_size, -1)
			frame_features[:, t, :] = self.fc1(feature)
		lstm_out, _ = self.lstm(frame_features)
		return self.fc2(lstm_out[:, -1, :])
			

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