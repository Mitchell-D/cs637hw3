""" Annotated torch code for the MNIST portion of cs637hw3 """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SaveOutput:
    """ Basic hook class for storing layer states """
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    def clear(self):
        self.outputs = []

class CNNNet(nn.Module):
    """ """
    def __init__(self):
        super(CNNNet, self).__init__()
        # Convolutional layers
        #Init_channels, channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
                )
        self.conv2 = nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
                )
        self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                )

        # Pooling layers
        self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                )

        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)

        # Dropout layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        0.  Input RGB: (M,N,3)
        1.  2d conv padded 1px with 16 of 3x3 kernels: (B,M-2,N-2,16)
        2.  exponential linear unit activation
        3.  2x2 max pool with stride of 2: (B,(M-2)/2,(N-2)/2,32)

        4.  2d conv padded 1px with 32 of 3x3 kernels: (B,(M-6)/2,(N-6)/2)
        5.  exponential linear unit activation
        6.  2x2 max pool with stride of 2: (B,(M-6)/4,(N-6)/4,32)

        7.  2d conv padded 1px with 64 of 3x3 kernels: (B,(N-14)/4,(N-14)/4,64)
        8.  exponential linear unit activation
        9.  2x2 max pool with stride of 2: (B,(M-6)/8,(N-6)/8,64)

        10. Flatten activations: (B,64*(M-6)(N-6)/64)
        11. Dropout (25%)
        12. Feedforward layer: (B, 500)
        13. Exponential linear unit activation
        14. Dropout (25%)
        15. Feedforward layer: (B, 10)
        """
        self.conv1_out = F.elu(self.conv1(x))
        self.conv2_out = F.elu(self.conv2(self.pool(self.conv1_out)))
        self.conv3_out = F.elu(self.conv3(self.pool(self.conv2_out)))

        # Flatten the image
        x = self.pool(self.conv3_out).view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        """
        Initialize the deep CNN's layers
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1
                )
        self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1
                )
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        ## 9216 = (16-4) * (16-4) * 64
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        0.  Input: (B, M, N, 1)
        1.  2d convolution with 32 of 3x3 kernels: (B, M-2, N-2, 32)
        2.  ReLU activation
        3.  2d convolution with 64 of 3x3 kernels: (B, M-4, N-4, 64)
        4.  ReLU activation
        5.  Max pool with 2x2 domain: (B, (M-4)/2, (N-4)/2, 64)
        6.  Dropout (25%)
        7.  Collapse the spatial dimensions: (B, 64*(M-4)(N-4)/4)
        8.  Feedforward layer: (B, 9216)
        9.  ReLU activation
        10. Drouput (50%)
        11. Feedforward layer: (B, 10)
        12. Softmax activation
        """
        #self.conv1_out = self.conv1(x)
        self.conv1_out = F.relu(self.conv1(x))
        self.conv2_out = F.relu(self.conv2(self.conv1_out))

        x = F.max_pool2d(self.conv2_out, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
