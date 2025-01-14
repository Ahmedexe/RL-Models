import torch.nn as nn
import torch
import random
import numpy as np
import torch.optim as  optim
from env import TicTacToe


stateSize = 3**9
actionSize = 9
bufferSize = 8000
batchSize = 32
gamma = 0.95
epsilon = 1.
epsilonDecay = 0.995
alpha = 0.003
episodes = 350
targetupdate = 128


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



env = TicTacToe()
state = env.reset()

policyNN = QNN(stateSize, actionSize)
targetNN = QNN(stateSize, actionSize)
targetNN.load_state_dict(policyNN.state_dict())
targetNN.eval()

optimizer = optim.Adam(policyNN.parameters(), lr=alpha)


for episode in range(episodes):

    if episode % targetupdate == 0:
        targetNN.load_state_dict(policyNN.state_dict())

    state = env.reset()
    totalReward = 0
    done = False

    while not done:

        # Epsilon greedy Exploration..
        if random.random() < epsilon:
            action = str(random.randint(0, 8))

        else:
            stateTensor = torch.tensor([float(x) for x in state])
            action = torch.argmax(policyNN(stateTensor)).item()


        ## Take Action ----> To Be Conti.