import torch.nn as nn

class PoorPerformingCNN(nn.Module):  # Correct inheritance from nn.Module
    def __init__(self):
        # Initialize the parent class (nn.Module)
        super(PoorPerformingCNN, self).__init__()  

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 4 channels
        self.relu1 = nn.ReLU()  # Activation function for non-linearity
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Second convolutional block
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)  # Input: 4 channels, Output: 8 channels
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(8 * 8 * 8, 10)  # Input: Flattened feature maps, Output: 10 classes

    def forward(self, x):
        # Pass input through the first convolutional block
        x = self.pool1(self.relu1(self.conv1(x)))

        # Pass through the second convolutional block
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layer
        x = self.fc1(x)
        return x
    
