# This is the early exiting model, which will be trained separately. 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check CUDA device
if torch.cuda.is_available():
    print(f'Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else:
    print('CUDA not available, using CPU.')

# Parameters
batch_size =  64
valid_size = 0.1
random_seed = 1
classes = [str(i) for i in range(10)]

# Transformations for training, validation, and test data
transform_train = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

transform_test = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Split training data into training and validation
num_train = len(train_dataset)
split = int(valid_size * num_train)
train_data, valid_data = random_split(train_dataset, [num_train - split, split], generator=torch.Generator().manual_seed(random_seed))

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define Early Exit model (3 Convolution, 2 FC)
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
