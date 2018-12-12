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
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from PIL import Image

from light_cnn import LightCNN_Layers
import cv2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--resume', default='F:/wflw/modelv1/modell1/lightCNN_5991_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='root path of face images (default: none).')
parser.add_argument('--save_path', default='', type=str, metavar='PATH', 
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=196, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')


def main():
    global args
    args = parser.parse_args()

    model = LightCNN_Layers(num_classes=args.num_classes)
    model.eval()
    if args.cuda:
        model=model.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])   
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    for name, value in model.named_parameters():
        print(name)
    print(model.conv_pre.weight)
    print(model.conv_pre.padding)
    print(model.conv_pre.kernel_size)
    print(model.conv_pre.stride)

    transform = transforms.Compose([transforms.ToTensor()])
    count     = 0
    input     = torch.zeros(1, 1, 112, 112)
    imgFile='F:/casia-maxpy-clean/webface144X144/0000156/005.jpg'
    opencvimg = cv2.imread(os.path.join(args.root_path, imgFile), cv2.IMREAD_GRAYSCALE)
   # opencvimg = opencvimg[6:112+6,6:112+6]
    opencvimg = cv2.resize(opencvimg,(112,112))
    #img = Image.fromarray(grayImg)
    img   = np.reshape(opencvimg, (112, 112, 1))
    img   = transform(img)
    input[0,:,:,:] = img

    start = time.time()
    if args.cuda:
        input_var = input.cuda()
        output = model(input_var)
        end  = time.time() - start
        landmarks = output.data.cpu().numpy()[0].tolist()
        for i in range(len(landmarks)//2):
            pointx = float(landmarks[i])
            pointy = float(landmarks[i+98])
            cv2.circle(opencvimg, (int(pointx), int(pointy)), 1, (255,255, 0), 2)
        cv2.imshow('opencvimg2.jpg', opencvimg)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
