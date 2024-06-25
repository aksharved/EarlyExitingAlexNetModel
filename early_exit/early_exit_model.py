# This is the early exiting model which will be trained separately.
import torch
import torch.nn as nn

# Define the Early Exit model with 3 convolution layers and 2 fully connected. 
class EarlyExit(nn.Module):
    def __init__(self, num_classes=10):
        super(EarlyExit, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.early_exit_fc1 = nn.Linear(384 * 13 * 13, 1024)
        self.early_exit_fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        early_exit_out = self.early_exit_fc1(out.reshape(out.size(0), -1))
        early_exit_out = self.early_exit_fc2(early_exit_out)
        return early_exit_out
