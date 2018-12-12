from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Wing_loss
import Gcs_loss
from collections import OrderedDict

import numpy as np

from light_cnn import LightCNN_Layers
from load_imglist import ImageList
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--multi_GPU', default=False,help="use multi-GPU")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=6000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='LightCNN_Layers', type=str, metavar='Model',
                    help='model type: LightCNN_Layers')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='../cropWFLWImgsTrain/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='../croptrain.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--save_path', default='../model/v18/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=196, type=int,
                    metavar='N', help='number of classes (default: 10)')

outLog = open('../log/wing1.5.txt','w')
def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'LightCNN_Layers':
        model = LightCNN_Layers(num_classes=args.num_classes)

    #wing loss: cvpr 2018
    criterion = Wing_loss.Wing_loss()
    # criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    criteriongcs = Gcs_loss.Gcs_loss()

    if args.cuda:
        print('use cuda')
        model = model.cuda()
        criterion = criterion.cuda()
        criteriongcs = criteriongcs.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                #weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.multi_GPU:
               new_state_dict = OrderedDict()
               for k, v in checkpoint['state_dict'].items():
                   name = k[7:]  # remove `module.`
                   new_state_dict[name] = v
               # load params
               model.load_state_dict(new_state_dict)
            else:

                model_dict = model.state_dict()

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)


                #model.load_state_dict(checkpoint['state_dict'])
    
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #load image
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list, 
            transform=transforms.Compose([ 
                transforms.ToTensor()
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, criteriongcs, optimizer, epoch)
        if epoch%10 == 0:
            save_name = args.save_path + 'lightCNN_' + str(epoch+1) + '_checkpoint.pth.tar'
            save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
        }, save_name)

mylambda = 1.5
def train(train_loader, model, criterion, criteriongcs, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()


    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var      = input.cuda()
        target_var     = target.cuda()
        #print(target_var.size())
        #print(target_var.size()[0])
        # compute output
        output = model(input_var)

        loss1   = criterion(output, target_var)
        loss2 = criteriongcs(output, target_var)

        #print(loss1)
        #print(loss2)
        loss = loss1 + mylambda * loss2



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} \t'
                  'Data {data_time.val:.3f} '.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time),end=' ')
            print('loss1: ' +  str(loss1.item()) + '\t' + 'loss2: ' + str(loss2.item()*mylambda))
            outLog.write('loss1: ' +  str(loss1.item()) + '\t' + 'loss2: ' + str(loss2.item()*mylambda))
            outLog.write('\n')
            outLog.flush()



def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step  = 50
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    main()
    outLog.close()
