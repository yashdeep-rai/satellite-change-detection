import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self,x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        return x


class ChangeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,1),
            nn.Sigmoid()
        )

    def forward(self,imgA,imgB):

        f1 = self.encoder(imgA)
        f2 = self.encoder(imgB)

        diff = torch.abs(f1 - f2)

        out = self.decoder(diff)

        return out
