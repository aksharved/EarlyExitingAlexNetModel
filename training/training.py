# We will train both of the models over 50 epochs in the same loop. 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
from pathlib import Path
from early_exit.early_exit_model import EarlyExit
from full_model.full_model import FullModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check CUDA device
if torch.cuda.is_available():
    print(f'Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else:
    print('CUDA not available, using CPU.')

# Parameters
batch_size = 64
num_epochs = 50
learning_rate = 0.001
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

# Ensure the saved_models directory exists
saved_models_dir = Path('saved_models')
saved_models_dir.mkdir(parents=True, exist_ok=True)

# Initialize both models
early_exit_model = EarlyExit(10).to(device)
full_model = FullModel(10).to(device)

# Define loss function and optimizers for both models
criterion = nn.CrossEntropyLoss()
optimizer_early_exit = torch.optim.Adam(early_exit_model.parameters(), lr=learning_rate)
optimizer_full = torch.optim.Adam(full_model.parameters(), lr=learning_rate)

# Training loop for both EarlyExit and FullModel
for epoch in range(num_epochs):
    early_exit_model.train()
    full_model.train()

    running_loss_early_exit = 0.0
    running_loss_full = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer_early_exit.zero_grad()
        optimizer_full.zero_grad()

        # Forward pass for early exit model
        outputs_early_exit = early_exit_model(images)
        loss_early_exit = criterion(outputs_early_exit, labels)

        # Forward pass for full model
        outputs_full = full_model(images)
        loss_full = criterion(outputs_full, labels)

        # Backward pass and optimization for early exit model
        loss_early_exit.backward()
        optimizer_early_exit.step()

        # Backward pass and optimization for full model
        loss_full.backward()
        optimizer_full.step()

        # Accumulate the losses
        running_loss_early_exit += loss_early_exit.item()
        running_loss_full += loss_full.item()

    # Calculate average losses
    avg_loss_early_exit = running_loss_early_exit / len(train_loader)
    avg_loss_full = running_loss_full / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Early Exit Model Loss: {avg_loss_early_exit:.4f}, Full Model Loss: {avg_loss_full:.4f}')

    # Validate the models
    early_exit_model.eval()
    full_model.eval()
    correct_early_exit = 0
    correct_full = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Early exit model validation
            outputs_early_exit = early_exit_model(images)
            _, predicted_early_exit = torch.max(outputs_early_exit.data, 1)
            correct_early_exit += (predicted_early_exit == labels).sum().item()

            # Full model validation
            outputs_full = full_model(images)
            _, predicted_full = torch.max(outputs_full.data, 1)
            correct_full += (predicted_full == labels).sum().item()

            total += labels.size(0)

    accuracy_early_exit = 100 * correct_early_exit / total
    accuracy_full = 100 * correct_full / total

    print(f'Validation Accuracy: Early Exit Model: {accuracy_early_exit:.2f}%, Full Model: {accuracy_full:.2f}%')

# Save the model weights
early_exit_model_path = 'saved_models/early_exit_model.pth'
full_model_path = 'saved_models/full_model.pth'
torch.save(early_exit_model.state_dict(), early_exit_model_path)
torch.save(full_model.state_dict(), full_model_path)
print(f"Early Exit Model saved to {early_exit_model_path}")
print(f"Full Model saved to {full_model_path}")

print('Finished Training')
