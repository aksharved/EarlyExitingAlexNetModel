# CIFAR-10

This repository contains the code for training and evaluating two machine learning models: an Early Exit Model and a Full Model. The goal is to increase accuracy by training these models separately and using an early exit mechanism based on a confidence threshold.

## Repository Structure

- `early_exit/`: Contains the code for the Early Exit Model.
  - `early_exit_model.py`: Defines the Early Exit Model architecture.
  - `training_early_exit.py`: Training script for the Early Exit Model.

- `full_model/`: Contains the code for the Full Model.
  - `full_model.py`: Defines the Full Model architecture.
  - `training_full_model.py`: Training script for the Full Model.

- `evaluation/`: Contains the evaluation script.
  - `evaluate_models.py`: Script to load the trained models and evaluate them using a confidence threshold for early exit.

- `data/`: Contains data-related scripts.
  - `download_data.py`: Script to download and preprocess the CIFAR-10 dataset.

- `saved_models/`: Directory to save and load trained model weights.

### Dataset: CIFAR-10

The CIFAR-10 dataset is a widely-used benchmark dataset in machine learning and computer vision. It consists of 60,000 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images. Each image is 32x32 pixels in size and falls into one of the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The CIFAR-10 dataset is used for training machine learning and computer vision algorithms, providing a challenging task for image classification due to the variety of objects and backgrounds.


#### Prerequisites

- Python 3.8+
- PyTorch
- Matplot

