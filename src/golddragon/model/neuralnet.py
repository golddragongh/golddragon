import torch
import torch.nn as nn

class TorchNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, 4)
        self.layer2 = nn.Linear(4, 3)
        self.layer3 = nn.Linear(3, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        z1 = self.layer1(x)
        a1 = self.relu(z1)

        z2 = self.layer2(a1)
        a2 = self.relu(z2)

        z3 = self.layer3(a2)

        return z3