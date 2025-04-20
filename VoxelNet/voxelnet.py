from torch import nn

class VoxNet(nn.Module):
    # input size: (batch_size, num_channels, 32, 32, 32)
    # in this instance num_channels = 1 since grayscale voxel grids, (32, 32, 32) is the size of the voxel grid
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # Convolutional layers
        # 3D convolutional layer. Input 1 channel -> output 32 channels, meaning 32 different kernels are applied to the input
        # and 32 feature maps will be obtained (those 32 filters parameters will be updated during training as would be for normal weights)
        # recalling that output_size = truncate((input_size - kernel_size + 2 * padding) / stride) + 1, so in this case choosing stride = 2 and padding = 3
        # results to (32 - 5 + 6) / 2 + 1 = 33/2 + 1 = 16 + 1 = 17
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=3)

        # input 32 channels, output 7 and 3x3x3 kernel. The input size is 17, so the output size will be (17 - 3) / 1 + 1 = 15
        self.conv2 = nn.Conv3d(32, 7, kernel_size=3, stride=1, padding=0)

        # convolution used for residual block (doesn't change the output size)
        self.convRes = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)

        # Pooling layer (3x3x3 kernel), stride is 2 by default
        self.pool = nn.MaxPool3d(3)

        # batch normalization per layer
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(7)
        self.bnRes = nn.BatchNorm3d(32)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 5 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, n_classes)

        # Misc
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # add non linearity. Same shape, simply puts negative values to 0

        # Residual layer
        shortcut = x.clone()

        x = self.convRes(x)
        x = self.bnRes(x)
        x = self.relu(x)

        x = shortcut + x  # this helps preserve gradient flow and improve training
        shortcut = x.clone()

        x = self.convRes(x)
        x = self.bnRes(x)
        x = self.relu(x)

        x = shortcut + x

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Pooling
        x = self.pool(x)
        
        # Flatten (multiply all dimensions to create a single dimension array)
        x = x.flatten(1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output
        x = self.fc2(x)

        return x