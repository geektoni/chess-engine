import torch.nn as nn
import torch.nn.functional as F


class ChessEngine(nn.Module):
    """
    This module contains the ConvNet models which estimates
    who is winning between White and Black.
    """

    def __init__(self):
        super(ChessEngine, self).__init__()

        self.conv1 = nn.Conv2d(8, 8, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 4, 2)
        self.fc1 = nn.Linear(4 * 1 * 2, 8)
        self.fc2 = nn.Linear(8, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):

        x = x.float()

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4 * 1 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x