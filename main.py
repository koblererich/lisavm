#!/usr/bin/env python3

from datetime import datetime
import os
import argparse
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

import utils
import models
from dataset_MNIST import MNIST
from dataset_CIFAR10 import CIFAR10
from dataset_TinyImageNet import TestTinyImageNetDataset, TrainTinyImageNetDataset


matplotlib.use("Agg")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default="mnist",
                        help='dataset "mnist" or "tiny-imagenet" (default: mnist)')
    parser.add_argument('--model', type=str, default="resnet188",
                        help='model used for tiny-imagenet (default: resnet188)')
    parser.add_argument('--optim', type=str, default="lisa",
                        choices=["lisa", "lisa-vm", "sgd", "adam", "adabelief"],
                        help='select optimizer')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--steps', type=int, default=10_000, metavar='N',
                        help='number of steps to train (default: 14)')
    parser.add_argument('--alpha', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Variance smoothing momentum (default: .9)')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='VM momentum (default: .9)')
    parser.add_argument('--beta3', type=float, default=0.999,
                        help='VM momentum 2nd (default: .999)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay') 
    parser.add_argument('--eps-k-fact', type=float, default=1.,
                        help='epsilon k sequence factor (default: 1.0)')
    parser.add_argument('--ls_ci', type=float, default=.8,
                        help='confidence interval for non-montone LS (default: 0.8)')
    parser.add_argument('--gamma1', type=float, default=100,
                        help='scaling factor of eps_k sequence for variance bound (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--acc', type=float, default=0.,
                        help='extrapolation factor')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--comment', type=str, default="")
    parser.add_argument('--log-interval', type=int, default=500, metavar='Nl',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=1_000, metavar='Nt',
                        help='how many batches to wait before testing status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 8,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset.lower() == "mnist":
        transform=transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = MNIST('./data', train=True, download=True, transform=transform)
        dataset2 = MNIST('./data', train=False, transform=transform)
        num_channels = 1
        num_classes = 10
    elif args.dataset.lower() == "cifar":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset1 = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        
        dataset2 = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_channels = 3
        num_classes = 10
    else:
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2764, 0.2689, 0.2816))
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=.3, contrast=.1, saturation=.1, hue=.1),
            normalize,
        ])
        dataset1 = TrainTinyImageNetDataset(transform=transform)
        dataset2 = TestTinyImageNetDataset(transform=normalize)
        num_channels = 3
        num_classes = 200

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = models.get(args.model, num_channels=num_channels, num_classes=num_classes, use_bn=False).to(device)

    optim = args.optim.lower()
    assert optim in ["lisa", "lisa-vm", "sgd", "adam", "adabelief"]
    tmp = optim
    if optim == "lisa-vm":
        tmp += f"_betas{args.beta1}-{args.beta2}-{args.beta3}"
    elif optim in ["adam", "adabelief"]:
        tmp += f"_betas{args.beta1}-{args.beta2}_acc{args.acc}"
    log_dir = os.path.join("runs-final", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + \
              f"_{args.dataset}_{args.model}_{tmp}_alpha{args.alpha:}_bs{args.batch_size}_wd{args.weight_decay}_{args.comment}"
    writer = SummaryWriter(log_dir=log_dir)

    if optim in ["lisa", "lisa-vm"]:
        utils.train_lisa(args, model, device, dataset1, test_loader, writer)
    else:
        utils.train_std(args, model, device, train_loader, test_loader, writer)

    print("\n Final testing")
    model.eval()
    utils.test(model, device, test_loader, writer, args.steps)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
