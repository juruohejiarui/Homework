# only adversarial attack
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import math
from tqdm import tqdm

def train(model : nn.Module, 
          epochs : int,
          learning_rate : float, 
          momentum : float,
          alpha : float,
          epsilon : float,
          k : int,
          train_data : DataLoader, 
          test_data : DataLoader,
          logger : SummaryWriter) :
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_data), 1e-7)

    for epoch in tqdm(range(epochs)) :
        acc_train = 0
        loss_train = 0
        acc_test = 0
        loss_test = 0
        for i, (x, y) in enumerate(train_data) :
            x : torch.Tensor = x.cuda()
            y : torch.Tensor = y.cuda()
            
            y_pred = model(x)

            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            model.zero_grad()

            # update acc and loss

            acc_train += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            loss_train += loss.sum().item()

        # test
        for i, (x, y) in enumerate(test_data) :
            x : torch.Tensor = x.cuda()
            y : torch.Tensor = y.cuda()
            delta : torch.Tensor = torch.zeros(x.shape).cuda()
            for j in range(k + 1) :
                x_a : torch.Tensor = (x + delta).cuda()
                x_a.requires_grad = True
                y_pred : torch.Tensor = model(x_a)

                loss = F.cross_entropy(y_pred, y)
                loss.backward()

                x_grad = x_a.grad.clone()
                
                delta = (delta + alpha * x_grad.norm()).clamp(-epsilon, epsilon)

                model.zero_grad()

            # update acc and loss
            acc_test += (y_pred.argmax(1) == y.argmax(1)).sum().item()
            loss_test += loss.sum().item()

        # log
        acc_train /= len(train_data.dataset)
        loss_train /= len(train_data.dataset)
        acc_test /= len(test_data.dataset)
        loss_test /= len(test_data.dataset)
        logger.add_scalar(f"train/acc", acc_train, epoch)
        logger.add_scalar(f"train/loss", loss_train, epoch)
        logger.add_scalar(f"test/acc", acc_test, epoch)
        logger.add_scalar(f"test/loss", loss_test, epoch)