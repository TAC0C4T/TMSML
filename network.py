import torch
import torch.nn as nn
import torch.optim as optim
from parse import parse

class Network(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(9):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Threshold(-0.055,0.04))

        # Final output layer (no ReLU)
        layers.append(nn.Linear(hidden_size, output_size))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)