# After running the first three files, we will evaluate the efficiency of this model with this code. 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from early_exit.early_exit_model import EarlyExit
from full_model.full_model import FullModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations for testing
transform_test = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

# Load CIFAR-10 test dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Load the saved model weights
early_exit_model_path = 'saved_models/early_exit_model.pth'
full_model_path = 'saved_models/full_model.pth'

early_exit_model = EarlyExit(10).to(device)
full_model = FullModel(10).to(device)

early_exit_model.load_state_dict(torch.load(early_exit_model_path))
full_model.load_state_dict(torch.load(full_model_path))

early_exit_model.eval()
full_model.eval()

# Testing loop with confidence threshold
with torch.no_grad():
    ee_counter = 0
    full_counter = 0
    threshold = 0.95 # if the EE model is 95 % sure or higher in its prediciton, the model will exit early. 
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Pass through early exit model
        early_exit_outputs = early_exit_model(images)
        softmax_outputs = F.softmax(early_exit_outputs, dim=1)

        # Find the maximum probabilities and their indices
        confidence, predicted_classes = torch.max(softmax_outputs, dim=1)

        # Iterate over each sample in the batch
        for i in range(images.size(0)):
            if confidence[i].item() > threshold:
                ee_counter += 1
                predicted = predicted_classes[i]
            else:
                full_counter += 1
                final_output = full_model(images[i].unsqueeze(0))
                _, predicted = torch.max(final_output, 1)

            n_samples += 1
            n_correct += (predicted == labels[i]).sum().item()

            label = labels[i].item()
            if label == predicted.item():
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    test_acc = 100.0 * n_correct / n_samples
    print(f'Test Accuracy of the network with confidence threshold {threshold}: {test_acc:.2f} %')
    print(f'Went through full model {full_counter} times')
    print(f'Early exit counter is {ee_counter}')
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')
