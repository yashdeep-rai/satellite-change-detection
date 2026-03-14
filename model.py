
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.net(x)

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DoubleConv(3,32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32,64)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x

class ChangeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNetEncoder()

        self.decoder = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,1),
            nn.Sigmoid()
        )

    def forward(self,a,b):

        f1 = self.encoder(a)
        f2 = self.encoder(b)

        diff = torch.abs(f1-f2)

        out = self.decoder(diff)

        out = F.interpolate(out,size=(256,256),mode="bilinear",align_corners=False)

        return out
