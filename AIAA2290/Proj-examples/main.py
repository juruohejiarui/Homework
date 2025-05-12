import torch
import models.cnn
import data.mnist
import pgd
import argparse
from tensorboardX import SummaryWriter

if __name__ == "__main__" :
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=50)
    parse.add_argument("--batchSize", type=int, default=64)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--num_iterate", type=int, default=40)
    parse.add_argument("--epsilon", type=float, default=0.3)
    parse.add_argument("--alpha", type=float, default=0.01)
    parse.add_argument("log", type=str)
    
    args = parse.parse_args()

    train_loader, test_loader = data.mnist.getLoader(batch_size=args.batchSize)
    model = models.cnn.CNN(in_channels=1, num_classes=10).cuda()
    logger = SummaryWriter(f"run/{args.log}", flush_secs=1)

    pgd.train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        num_iterate=args.num_iterate,
        num_attack_iterate=[0,10,40],
        epsilon=args.epsilon,
        alpha=args.alpha,
        logger=logger,
        model_name=args.log
    )