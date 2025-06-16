import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    A simple 5‐layer convolutional network followed by two fully connected layers.
    Input: 1×224×224 grayscale image
    Output: num_classes logistic probabilities (sigmoid)
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After all pool layers, the 224×224 input shrinks:
        # 224→112→56→28→14→7 (so feature map is 192×7x7 after conv5+pool).
        # We flatten that into a vector for the FC layers.
        flattened_size = 192 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolution + ReLU + MaxPool (×4)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)

        # Flatten and feed through FC layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))

        # Use sigmoid because this is a multi‐label problem (each class is independent)
        x = torch.sigmoid(self.fc2(x))
        return x
