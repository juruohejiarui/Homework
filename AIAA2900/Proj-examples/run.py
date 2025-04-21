import data
import data.mnist
import models
import models.resnet
import argparse
import freeat
import pgd
import pgdattack
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--momentum", type=int, default=0.78)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--iterate", type=int, default=5)
parser.add_argument("name", type=str)

if __name__ == "__main__" :
    args = parser.parse_args()

    (train_loader, test_loader) = data.mnist.getLoader(args.batch_size)
    model = models.resnet.ResNet18(num_classes=10, input_channel=1).cuda()
    logger = SummaryWriter(f"./run/{args.name}", flush_secs=10)
    
    pgdattack.train(model, 
                 epochs=args.epochs, 
                 learning_rate=args.lr, 
                 momentum=args.momentum, 
                 epsilon=10/255,
                 alpha=5/255, 
                 k=args.iterate, 
                 train_data=train_loader, 
                 test_data=test_loader,
                 logger=logger)