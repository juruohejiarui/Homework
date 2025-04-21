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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_data), 1e-7)

    for epoch in tqdm(range(epochs // k)) :
        acc_train = [0] * k
        loss_train = [0] * k
        acc_test = [0] * k
        loss_test = [0] * k
        for i, (x, y) in enumerate(train_data) :
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
                
                delta = (delta + alpha * x_grad.sign()).clamp(-epsilon, epsilon)

                optimizer.step()
                scheduler.step()

                # update acc and loss
                acc_train[j] += (y_pred.argmax(1) == y.argmax(1)).sum().item()
                loss_train[j] += loss.sum().item()

        # test
        for i, (x, y) in enumerate(test_data) :
            x : torch.Tensor = x.cuda()
            y : torch.Tensor = y.cuda()
            delta : torch.Tensor = torch.zeros(x.shape).cuda()
            for j in range(k) :
                x_a : torch.Tensor = (x + delta).cuda()
                x_a.requires_grad = True
                y_pred : torch.Tensor = model(x_a)

                loss = F.cross_entropy(y_pred, y)
                loss.backward()

                x_grad = x_a.grad.clone()
                
                delta = (delta + alpha * x_grad.norm()).clamp(-epsilon, epsilon)
                
                # update acc and loss

                acc_test[j] += (y_pred.argmax(1) == y.argmax(1)).sum().item()
                loss_test[j] += loss.sum().item()

                model.zero_grad()

        # log
        for i in range(k) :
            acc_train[i] /= len(train_data.dataset)
            loss_train[i] /= len(train_data.dataset)
            acc_test[i] /= len(test_data.dataset)
            loss_test[i] /= len(test_data.dataset)
            logger.add_scalar(f"train/acc_{i}", acc_train[i], epoch)
            logger.add_scalar(f"train/loss_{i}", loss_train[i], epoch)
            logger.add_scalar(f"test/acc_{i}", acc_test[i], epoch)
            logger.add_scalar(f"test/loss_{i}", loss_test[i], epoch)