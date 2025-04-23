from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torchvision.transforms

class SelfDataSet :
    def __init__(self, train) :
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, ), (0.5, ))
        ])
        self.data = MNIST(root="./dataset", train=train, download=True, transform=transform)
    def __len__(self) : 
        return len(self.data)
    def __getitem__(self, idx) :
        x, y = self.data[idx]
        y_onehot = torch.zeros(10)
        y_onehot[y] = 1
        return (x, y_onehot)

def getLoader(batch_size : int) :
    train_data = SelfDataSet(True)
    test_data = SelfDataSet(False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    return (train_loader, test_loader)