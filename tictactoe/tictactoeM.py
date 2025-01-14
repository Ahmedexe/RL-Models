import torch.nn as nn
import torch
import random
import numpy as np
import torch.optim
import fun

stateSize = 3**9
actionSize = 9
bufferSize = 2000
batchSize = 32
gamma = 0.95
epsilon = 1.
epsilonDecay = 0.995
alpha = 0.003
episodes = 250

fun.winPos(2)


class QNN(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(stateSize, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, actionSize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
