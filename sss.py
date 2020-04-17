from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
from optimizer import *

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--sparsity-structure-selection', '-sss', dest='sss', action='store_true',
                        help='train with sparsity-structure-selection')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        metavar='G', help='gamma (default: 0.01)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='ResNet20_cifar', type=str,
                        help='architecture to use')

    args = parser.parse_args()
    return args
args = get_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/yangke_huang/data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/home/yangke_huang/data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/home/yangke_huang/data/cifar100', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                          ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/home/yangke_huang/data/cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch]()
if args.cuda:
    model.cuda()
print(model)

optimizer1 = SGD([{'params': model.parameters()}], lr=args.lr,
                 momentum=args.momentum, weight_decay=args.weight_decay)
if args.sss:
    optimizer2 = APGNAG([{'params': model.lambda_block}],
                        lr=args.lr, momentum=args.momentum, gamma=args.gamma)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        #args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer1.load_state_dict(checkpoint['optimizer1'])
        if 'optimizer2' in checkpoint.keys():
            optimizer2.load_state_dict(checkpoint['optimizer2'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    print(model.lambda_block)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if args.sss:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            output = model.forward_sss(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            r1_loss = args.gamma * F.l1_loss(model.lambda_block, torch.zeros(model.lambda_block.size()).cuda(), reduction='sum')
            r1_loss.backward()
            optimizer1.step()
            optimizer2.step()
        else:
            optimizer1.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            optimizer1.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if args.sss:
            output = model.forward_sss(data)
        else: output = model.forward(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct.item() / len(test_loader.dataset)

def test_time():
    import time
    model.eval()
    test_loss = 0
    correct = 0
    duration = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == 100: break
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        start_time = time.time()
        output = model.forward_test(data)
        duration += time.time() - start_time
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest Time: {}\n'.format(duration / 100))

if args.test:
    from thop import profile
    input = torch.randn(1, 3, 32, 32)
    if args.cuda: input = input.cuda()
    flops, params = profile(model, inputs=(input,))
    print("flops: ", flops, ", params: ", params)
    test_time()
    import sys
    sys.exit(0)

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer1.param_groups:
            param_group['lr'] *= 0.1
        if args.sss:
            for param_group in optimizer2.param_groups:
                param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    if args.sss:
        print("lambda '{}'\n".format(model.lambda_block))
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if args.sss:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
        }, is_best, filepath=args.save)
    else:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer1': optimizer1.state_dict(),
        }, is_best, filepath=args.save)

print("Best accuracy: " + str(best_prec1))