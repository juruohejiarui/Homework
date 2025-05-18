import torch
import dataloader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN, FocalLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

def test(model, test_data : DataLoader, criterion) :
	model.eval()
	test_loss = 0
	correct = 0
	tot = 0
	with torch.no_grad() :
		for data, target in test_data :
			data : torch.Tensor = data.cuda(); target : torch.Tensor = target.cuda()
			output : torch.Tensor = model(data)
			test_loss += criterion(output, target).sum().item()
			pred = output.argmax(1)
			correct += (pred == target.argmax(1)).sum().item()
			tot += target.shape[0]
	test_loss /= tot
	correct /= tot
	return test_loss, correct

def train(model : nn.Module, 
		  train_data : DataLoader, test_data : DataLoader, learning_rate, epochs, logger : SummaryWriter, fig_path) :
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.78)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.5e-3)
	# criterion = nn.CrossEntropyLoss()
	criterion = FocalLoss()

	train_loss = []
	train_acc = []
	test_loss = []
	test_acc = []

	for epoch in tqdm(range(epochs)) :
		model.train()
		epoch_loss = 0
		correct = 0
		tot = 0


		for data, target in train_data :
			data : torch.Tensor = data.cuda(); target : torch.Tensor = target.cuda()

			optimizer.zero_grad()
			output : torch.Tensor = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.sum().item()
			correct += (output.argmax(1) == target.argmax(1)).sum().item()
			tot += target.shape[0]

		epoch_loss /= tot
		correct /= tot

		test_loss_, test_acc_ = test(model, test_data, criterion)

		# print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {correct:.4f}, Test Loss: {test_loss_:.4f}, Test Acc: {test_acc_:.4f}")

		train_loss.append(epoch_loss)
		train_acc.append(correct)
		test_loss.append(test_loss_)
		test_acc.append(test_acc_)

		logger.add_scalar("train/loss", epoch_loss, epoch + 1)
		logger.add_scalar("train/acc", correct, epoch + 1)
		logger.add_scalar("test/loss", test_loss_, epoch + 1)
		logger.add_scalar("test/acc", test_acc_, epoch + 1)

	fig, ax = plt.subplots(2, 1, figsize=(10, 8))
	ax[0].plot(range(1, epochs + 1), train_loss, label='Train Loss')
	ax[0].plot(range(1, epochs + 1), test_loss, label='Test Loss')
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('Loss')

	ax[0].legend()
	ax[1].plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
	ax[1].plot(range(1, epochs + 1), test_acc, label='Test Accuracy')
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('Accuracy')
	ax[1].legend()

	plt.savefig(fig_path, dpi=300)

if __name__ == "__main__" :
	model = CNN().cuda()
	train_data = DataLoader(dataloader.DataSet("./Data", "train"), batch_size=100, shuffle=True, num_workers=2, drop_last=False)
	test_data = DataLoader(dataloader.DataSet("./Data", "test"), batch_size=100, shuffle=True, num_workers=2, drop_last=False)
	logger = SummaryWriter(f"./run/CNN-test", flush_secs=2)

	train(model, train_data, test_data, 3e-4, 200, logger, "CNN-test.png")