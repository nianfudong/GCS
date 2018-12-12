import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class landmarknet(nn.Module):
    def __init__(self, num_classes=196): #98 points
        super(landmarknet, self).__init__()
        self.conv_pre = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.prelu_pre = nn.PReLU(16)
        self.pool_pre = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv1_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.prelu_11 = nn.PReLU(16)
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.prelu_12 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.prelu_21 = nn.PReLU(32)
        self.conv2_2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.prelu_22 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.prelu_31 = nn.PReLU(64)
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.prelu_32 = nn.PReLU(128)
        self.ip1 = nn.Linear(7*7*128, 128)
        self.prelu_ip1 = nn.PReLU(128)
        self.ip2 = nn.Linear(128, 128)
        self.prelu_ip2 = nn.PReLU(128)
        self.ip3 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.pool_pre(self.prelu_pre(self.conv_pre(x)))
        x = self.prelu_11(self.conv1_1(x))
        x = self.pool1(self.prelu_12(self.conv1_2(x)))
        x = self.prelu_21(self.conv2_1(x))
        x = self.pool2(self.prelu_22(self.conv2_2(x)))
        x = self.prelu_31(self.conv3_1(x))
        #print(x.shape)
        x = self.prelu_32(self.conv3_2(x))

        x = x.view(x.size(0), -1)
        x = self.prelu_ip1(self.ip1(x))
        #print(x)
        x = self.prelu_ip2(self.ip2(x))
        x = self.ip3(x)
        return x

def LightCNN_Layers(**kwargs):
    model = landmarknet(**kwargs)
    return model

