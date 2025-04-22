import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.perturb as perturb
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pgdattack

def train(
        model : nn.Module,
        train_loader : DataLoader,
        test_loader : DataLoader,
        epochs : int,
        lr : float,
        num_iterate : int,
        epsilon : float,
        alpha : float,
        logger : SummaryWriter
) :
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)) :

        acc_train, loss_train = 0, 0
        acc_test, loss_test = 0, 0

        for i, (x, y) in enumerate(train_loader) :
            x : torch.Tensor = x.cuda()
            y : torch.Tensor = y.cuda()
            x_a = x.clone()
            for k in range(num_iterate) :
                x_a = perturb.preturb(model, x, x_a, y, epsilon=epsilon, alpha=alpha, val_max=1, val_min=0)
            model.train()
            y_pred : torch.Tensor = model(x_a, eval=False)
            loss = F.cross_entropy(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_train += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            loss_train += loss.sum().item()

        acc_train /= len(train_loader) * train_loader.batch_size
        loss_train /= len(train_loader) * (num_iterate + 1)

        logger.add_scalar(f"train/acc", acc_train, epoch + 1)
        logger.add_scalar(f"train/loss", loss_train, epoch + 1)

        pgdattack.test(model, epoch + 1, test_loader, epsilon=epsilon, alpha=alpha, num_iterate=num_iterate, logger=logger, logger_tag="test")
        pgdattack.test(model, epoch + 1, test_loader, epsilon=epsilon, alpha=alpha, num_iterate=0, logger=logger, logger_tag="test-original")