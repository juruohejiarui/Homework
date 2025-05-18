import torch
import dataloader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTMCNN, FocalLoss
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics
import numpy as np

def test(model, test_data : DataLoader, criterion) -> tuple[float, float, float, float, float]:
	model.eval()
	test_loss = 0
	correct = 0
	f1_tot = 0
	precision_tot, recall_tot = 0, 0
	tot = 0

	# store y_true and y_pred for confusion matrix
	y_true = []
	y_pred = []
	with torch.no_grad() :
		for data, target in test_data :
			data : torch.Tensor = data.cuda(); target : torch.Tensor = target.cuda()
			output : torch.Tensor = model(data)
			test_loss += criterion(output, target).sum().item()

			correct += (output.argmax(1) == target.argmax(1)).sum().item()
			tot += target.shape[0]

			y_true.append(target.argmax(1).cpu().numpy())
			y_pred.append(output.argmax(1).cpu().numpy())

	test_loss /= tot
	correct /= tot

	y_true = np.concatenate(y_true)
	y_pred = np.concatenate(y_pred)
	f1_tot = metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)
	precision_tot = metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
	recall_tot = metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)

	return test_loss, correct, f1_tot, precision_tot, recall_tot

def train(model : nn.Module, 
		  train_data : DataLoader, test_data : DataLoader, learning_rate, epochs, logger : SummaryWriter, fig_path) :
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.78)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	# criterion = nn.CrossEntropyLoss()
	criterion = FocalLoss()

	train_loss = []
	train_acc = []
	train_f1 = []
	train_precision = []
	train_recall = []

	test_loss = []
	test_acc = []
	test_f1 = []
	test_precision = []
	test_recall = []

	for epoch in tqdm(range(epochs)) :
		model.train()
		epoch_loss = 0
		correct = 0
		f1_tot = 0
		precision_tot, recall_tot = 0, 0
		tot = 0

		y_true = []
		y_pred = []

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

			y_true.append(target.argmax(1).cpu().numpy())
			y_pred.append(output.argmax(1).cpu().numpy())

		epoch_loss /= tot
		correct /= tot
		
		y_true = np.concatenate(y_true)
		y_pred = np.concatenate(y_pred)

		f1_tot = metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)
		precision_tot = metrics.precision_score(y_true, y_pred, average='macro', zero_division=1)
		recall_tot = metrics.recall_score(y_true, y_pred, average='macro', zero_division=1)

		test_loss_, test_acc_, test_f1_, test_precision_, test_recall_ = test(model, test_data, criterion)

		# print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {correct:.4f}, Test Loss: {test_loss_:.4f}, Test Acc: {test_acc_:.4f}")

		train_loss.append(epoch_loss)
		train_acc.append(correct)
		train_f1.append(f1_tot)
		train_precision.append(precision_tot)
		train_recall.append(recall_tot)

		test_loss.append(test_loss_)
		test_acc.append(test_acc_)
		test_f1.append(test_f1_)
		test_precision.append(test_precision_)
		test_recall.append(test_recall_)


		logger.add_scalar("train/loss", epoch_loss, epoch + 1)
		logger.add_scalar("train/acc", correct, epoch + 1)
		logger.add_scalar("train/f1", f1_tot, epoch + 1)
		logger.add_scalar("train/precision", precision_tot, epoch + 1)
		logger.add_scalar("train/recall", recall_tot, epoch + 1)

		logger.add_scalar("test/loss", test_loss_, epoch + 1)
		logger.add_scalar("test/acc", test_acc_, epoch + 1)
		logger.add_scalar("test/f1", test_f1_, epoch + 1)
		logger.add_scalar("test/precision", test_precision_, epoch + 1)
		logger.add_scalar("test/recall", test_recall_, epoch + 1)

	fig, ax = plt.subplots(1, 5, figsize=(20, 8))
	ax[0].plot(range(1, epochs + 1), train_loss, label='Train Loss')
	ax[0].plot(range(1, epochs + 1), test_loss, label='Test Loss')
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('Loss')
	ax[0].legend()

	ax[1].plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
	ax[1].plot(range(1, epochs + 1), test_acc, label='Test Accuracy')
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('Accuracy')
	ax[1].set_ylim(0.5, 1)
	ax[1].legend()

	ax[2].plot(range(1, epochs + 1), train_f1, label='Train F1 Score')
	ax[2].plot(range(1, epochs + 1), test_f1, label='Test F1 Score')
	ax[2].set_xlabel('Epochs')
	ax[2].set_ylabel('F1 Score')
	ax[2].set_ylim(0.5, 1)
	ax[2].legend()

	ax[3].plot(range(1, epochs + 1), train_precision, label='Train Precision')
	ax[3].plot(range(1, epochs + 1), test_precision, label='Test Precision')
	ax[3].set_xlabel('Epochs')
	ax[3].set_ylabel('Precision')
	ax[3].set_ylim(0.5, 1)

	ax[3].legend()

	ax[4].plot(range(1, epochs + 1), train_recall, label='Train Recall')
	ax[4].plot(range(1, epochs + 1), test_recall, label='Test Recall')
	ax[4].set_xlabel('Epochs')
	ax[4].set_ylabel('Recall')
	ax[4].set_ylim(0.5, 1)

	ax[4].legend()

	plt.savefig(fig_path, dpi=300)

if __name__ == "__main__" :
	model = LSTMCNN().cuda()
	train_data = DataLoader(dataloader.DataSet("./data", "train"), batch_size=100, shuffle=True, num_workers=2, drop_last=False)
	test_data = DataLoader(dataloader.DataSet("./data", "test"), batch_size=100, shuffle=True, num_workers=2, drop_last=False)
	logger = SummaryWriter(f"./run/CNN-test-3", flush_secs=2)

	train(model, train_data, test_data, 1e-3, 450, logger, "CNN-test-3.png")