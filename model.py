import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessEngine(nn.Module):
    """
    This module contains the ConvNet models which estimates
    who is winning between White and Black.
    """

    def __init__(self, encoding_type="one_hot"):
        super(ChessEngine, self).__init__()

        self.encoding = 2 if encoding_type == "one_hot" else 1

        self.conv1 = nn.Conv2d(8, 8, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 2)
        self.fc1 = nn.Linear(8 * 1 * self.encoding, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 1 * self.encoding)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x