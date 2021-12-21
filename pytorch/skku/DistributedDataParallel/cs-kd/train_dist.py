from __future__ import print_function

import argparse
import csv
import os, logging
# from math.fvcore.fvcore.nn.jit_handles import batchnorm_flop_jit

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler

import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset, CustomDataset

import torch.distributed as dist

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

parser = argparse.ArgumentParser(description='CS-KD Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type (32x32: CIFAR_ResNet18, CIFAR_DenseNet121, 224x224: resnet18, densenet121)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='cifar100', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
parser.add_argument('--dataroot', default='~/data/', type=str, help='data directory')
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--cls', '-cls', action='store_true', help='adding cls loss')
parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')

parser.add_argument("--local_rank", default=3, type=int)
parser.add_argument("--world_size", default=4, type=int)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend="nccl")
# Data
print('==> Preparing dataset: {}'.format(args.dataset))
if not args.cls:
    # trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)
    trainloader = CustomDataset('/home/data/Imagenet/')
    ddp = torch.utils.data.distributed.DistributedSampler(trainloader)
    trainloader = torch.utils.data.DataLoader(trainloader, batch_size=128, sampler=ddp, num_workers=2)
    # trainloader = torch.utils.data.DataLoader(trainloader, batch_size=128,shuffle=True, num_workers=8)
    valloader = CustomDataset('/home/data/Imagenet/', False)
    valloader = torch.utils.data.DataLoader(valloader, batch_size=128, sampler=ddp, num_workers=2)
else:
    trainloader = CustomDataset('/home/data/Imagenet/')
    ddp = torch.utils.data.distributed.DistributedSampler(trainloader)
    trainloader = torch.utils.data.DataLoader(trainloader, batch_size=128,shuffle=False,sampler=ddp, num_workers=2)
    valloader = CustomDataset('/home/data/Imagenet/', False)
    valloader = torch.utils.data.DataLoader(valloader, batch_size=128, sampler=ddp, num_workers=2)
    # trainloader, valloader = load_dataset(args.dataset, args.dataroot, 'pair', batch_size=args.batch_size)


num_class = trainloader.dataset.num_classes
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))

# Model
print('==> Building model: {}'.format(args.model))

net = models.load_model(args.model, num_class)
# print(net)

if use_cuda:
   
    net = net.cuda(args.local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    print(torch.cuda.device_count())
    print('Using CUDA..')

# if args.ngpu > 1:
#     # net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
#     net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

# logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
# set_logging_defaults(logdir, args)
# logger = logging.getLogger('main')
# logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

criterion = nn.CrossEntropyLoss()


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

kdloss = KDLoss(args.temp)

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     train_cls_loss = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()

#         batch_size = inputs.size(0)
#         print(batch_size)
        
#         if not args.cls:
#             outputs = net(inputs)
#             loss = torch.mean(criterion(outputs, targets))
#             train_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += targets.size(0)
#             correct += predicted.eq(targets.data).sum().float().cpu()
#         else:
#             targets_ = targets[:batch_size//2]
#             outputs = net(inputs[:batch_size//2])
#             loss = torch.mean(criterion(outputs, targets_))
#             train_loss += loss.item()

#             with torch.no_grad():
#                 outputs_cls = net(inputs[batch_size//2:])
#             cls_loss = kdloss(outputs, outputs_cls.detach())
#             loss += args.lamda * cls_loss
#             train_cls_loss += cls_loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += targets_.size(0)
#             correct += predicted.eq(targets_.data).sum().float().cpu()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         progress_bar(batch_idx, len(trainloader),
#                      'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f '
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_cls_loss/(batch_idx+1)))

#     logger = logging.getLogger('train')
#     logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}]'.format(
#         epoch,
#         train_loss/(batch_idx+1),
#         train_cls_loss/(batch_idx+1),
#         100.*correct/total))

#     return train_loss/batch_idx, 100.*correct/total, train_cls_loss/batch_idx

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_cls_loss = 0

    for batch_idx, (inputs1,inputs2, targets1,targets2) in enumerate(trainloader):
        
        if use_cuda:
            inputs1, inputs2, targets1, targets2 = inputs1.cuda(args.local_rank), inputs2.cuda(args.local_rank), targets1.cuda(args.local_rank), targets2.cuda(args.local_rank)
            
        batch_size = inputs1.size(0)

        if not args.cls:
            loss=0
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            loss += torch.mean(criterion(outputs1, targets1))
            loss += torch.mean(criterion(outputs2, targets2))
            loss = loss/2
            train_loss += loss.item()

            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            total += targets1.size(0)
            total += targets2.size(0)
            correct += predicted1.eq(targets1.data).sum().float().cpu()
            correct += predicted2.eq(targets2.data).sum().float().cpu()
        else:
            # targets_ = targets[:batch_size//2]
            outputs = net(inputs1)
            loss = torch.mean(criterion(outputs, targets1))
            train_loss += loss.item()

            with torch.no_grad():
                outputs_cls = net(inputs2)
            cls_loss = kdloss(outputs, outputs_cls.detach())
            loss += args.lamda * cls_loss
            train_cls_loss += cls_loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets1.size(0)
            correct += predicted.eq(targets1.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_cls_loss/(batch_idx+1)))

    # logger = logging.getLogger('train')
    # logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}]'.format(
    #     epoch,
    #     train_loss/(batch_idx+1),
    #     train_cls_loss/(batch_idx+1),
    #     100.*correct/total))
    
    dist.barrier()

    return train_loss/batch_idx, 100.*correct/total, train_cls_loss/batch_idx

def val(epoch):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating

    loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    # logger = logging.getLogger('val')
    # logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
    #     epoch,
    #     val_loss/(batch_idx+1),
    #     acc))

    if acc > best_val:
        best_val = acc
        checkpoint(acc, epoch)

    return (val_loss/(batch_idx+1), acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc, train_cls_loss = train(epoch)
    val_loss, val_acc = val(epoch)
    adjust_learning_rate(optimizer, epoch)

print("Best Accuracy : {}".format(best_val))
# logger = logging.getLogger('best')
# logger.info('[Acc {:.3f}]'.format(best_val))
