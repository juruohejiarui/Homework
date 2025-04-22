import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module) :
    def __init__(self, in_channels, num_classes) :
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten(start_dim=1)

        self.linear1 = nn.Linear(7 * 7 * 64, 1024)
        self.linear2 = nn.Linear(1024, num_classes)

    def forward(self, x : torch.Tensor, eval=False) :
        if eval :
            self.eval()
        else :
            self.train()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = F.relu(x)
        
        self.train()
        x = self.linear2(x)

        return x

    def first_layer(self) : return nn.Sequential(*[self.conv1, nn.ReLU(), self.pool1])
    def forward_excepts_first_layer(self, x : torch.Tensor, eval=False) :
        if eval :
            self.eval()
        else :
            self.train()

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)

        self.train()
        x = self.linear2(x)
        return x