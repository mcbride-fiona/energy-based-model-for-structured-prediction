import torch.nn as nn
import torch.nn.functional as F
from config import ALPHABET_SIZE

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Conv2d(64, ALPHABET_SIZE, kernel_size=(16, 3), stride=(1, 2))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.classifier(x)
        x = x.squeeze(2).transpose(1, 2)
        return x
