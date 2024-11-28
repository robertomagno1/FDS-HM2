import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: 3 channels, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization for stability
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions by 2

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input: 32, Output: 64 channels
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  # Input: Flattened feature maps, Output: 256
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)  # Input: 256, Output: 128
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 10)  # Output: 10 classes (e.g., CIFAR-10)

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers with dropout
        x = self.dropout(self.fc1_relu(self.fc1(x)))
        x = self.dropout(self.fc2_relu(self.fc2(x)))
        x = self.fc3(x)

        return x
