import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.perturb as perturb
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

def test(
		model : nn.Module,
		epoch_id : int,
		test_loader : DataLoader,
		epsilon : float,
		alpha : float,
		num_iterate : int,
		logger : SummaryWriter,
		logger_tag : str,
		example_root : str | None = None
) :
	acc_test, loss_test = 0, 0

	with torch.no_grad() :
		for i, (x, y) in enumerate(test_loader) :
			x : torch.Tensor = x.cuda()
			y : torch.Tensor = y.cuda()
			x_a = x.clone()

			for k in range(num_iterate) :
				x_a = perturb.preturb(model, x, x_a.data, y, epsilon=epsilon, alpha=alpha, val_max=1, val_min=0)

			y_pred : torch.Tensor = model(x_a, eval=True)
			loss = F.cross_entropy(y_pred, y)

			acc_test += (y_pred.argmax(1) == y.argmax(1)).sum().item()
			loss_test += loss.sum().item()

	acc_test /= len(test_loader) * test_loader.batch_size
	loss_test /= len(test_loader)
	logger.add_scalar(f"{logger_tag}/loss", loss_test, epoch_id)
	logger.add_scalar(f"{logger_tag}/acc", acc_test, epoch_id)