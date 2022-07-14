"""
Adapted from https://www.kaggle.com/code/khoongweihao/resnet-34-pytorch-starter-kit/notebook
"""

from torch import nn, flatten
import torch.nn.functional as F
from torchvision.models import resnet18



class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x





class ResNet18(nn.Module):
    def __init__(self, output_sigmoid):
        super(ResNet18,self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(7,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlock(64,256),
            ResidualBlock(256,256,2)
        )
        
        self.avgpool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(1024,9)

        self.output_sigmoid = output_sigmoid
        if self.output_sigmoid:
            self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)

        x = self.fc1(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)

        return x





def get_resnet18_pytorch(in_channels=7):

    net = resnet18(pretrained=False)

    net.conv1 = nn.Conv2d(in_channels, 
                          net.conv1.out_channels, 
                          kernel_size=net.conv1.kernel_size, 
                          stride=net.conv1.stride, 
                          padding=net.conv1.padding,bias=False)
    #import pdb; pdb.set_trace()
    net.layer2[0].conv1.stride = (1,1)
    net.layer2[0].downsample[0].stride = (1,1)
    net.layer3[0].conv1.stride = (1,1)
    net.layer3[0].downsample[0].stride = (1,1)
    net.layer4[0].conv1.stride = (1,1)
    net.layer4[0].downsample[0].stride = (1,1)

    net.fc = nn.Linear(512, out_features=9, bias = True)

    return net