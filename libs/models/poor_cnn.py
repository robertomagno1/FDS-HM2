import torch.nn as nn

class PoorPerformingCNN(nn.Module):  # Correct inheritance from nn.Module
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()  # Explicitly call the parent class constructor

        # Define the layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * 8 * 8, 10)  # Flattened output size from conv layers

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected
        x = self.fc1(x)
        return x