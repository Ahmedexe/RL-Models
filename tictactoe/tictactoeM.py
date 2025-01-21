import torch.nn as nn
import torch
import random
import numpy as np
import torch.optim as optim
from env import TicTacToe
import copy

np.random.seed(42)

stateSize = 9
actionSize = 9
bufferSize = 10000
batchSize = 256
gamma = 0.95
epsilon = 1.
epsilonDecay = 0.995
alpha = 0.00005
episodes = 50000
targetupdate = 512
experiences = []


class QNN(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(QNN, self).__init__()
        self.fc1 = nn.Linear(stateSize, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, actionSize)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


env = TicTacToe()
state = env.reset(0)

policyNN = QNN(stateSize, actionSize)
targetNN = QNN(stateSize, actionSize)
policyNN.load_state_dict(torch.load("tictactoeM2.pth"))
targetNN.load_state_dict(policyNN.state_dict())
targetNN.eval()

optimizer = optim.Adam(policyNN.parameters(), lr=alpha)
wins = 0

for episode in range(episodes):

    if episode % targetupdate == 0:
        targetNN.load_state_dict(policyNN.state_dict())

    state = env.reset(2)
    totalReward = 0
    done = False

    if len(experiences) > bufferSize:
        experiences.pop(0)

    while not done:

        # Epsilon greedy Exploration..
        if np.random.rand() < epsilon:
            action = np.random.randint(8)

        else:
            stateTensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = int(torch.argmax(policyNN(stateTensor)).item())

        state, nextState, reward, done = env.play(action, 1)
        experiences.append((state, action, reward, nextState, done))
        state = copy.deepcopy(nextState)
        totalReward += reward

        if (reward > 0):
            wins += 1

        # Second player plays
        with torch.no_grad():
            stateTensor = torch.tensor(state, dtype=torch.float32)
            random.seed(42)
            partialReward = 0
            # while partialReward < 1:
            #     _, state, partialReward, done = env.play(
            #         random.randint(0, 8), 2)
            state, nextState, reward2, done = env.play(
                int(torch.argmax(policyNN(stateTensor)).item()), 2)
            experiences.append((state, action, reward2, nextState, done))
            state = copy.deepcopy(nextState)

            if (reward2 < -1):
                state = env.playValidMove(2)

        if len(experiences) >= batchSize:
            expSample = random.sample(experiences, batchSize)
            stateSample, actionSample, rewardSample, nextStateSample, doneSample = zip(
                *expSample)

            stateSample = torch.tensor(stateSample, dtype=torch.float32)
            actionSample = torch.tensor(actionSample, dtype=torch.int64)
            rewardSample = torch.tensor(rewardSample, dtype=torch.float32)
            nextStateSample = torch.tensor(
                nextStateSample, dtype=torch.float32)
            doneSample = torch.tensor(doneSample, dtype=torch.float32)

            Qvals = policyNN(stateSample).gather(
                1, actionSample.unsqueeze(1)).squeeze(1)

            nextQvals = targetNN(nextStateSample).max(1)[0].detach()
            targetQs = rewardSample + (gamma * nextQvals * (1-doneSample))

            # Update QNN params
            loss = nn.MSELoss()(Qvals, targetQs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(0.1, epsilonDecay * epsilon)

        print(f"Episode {episode + 1}: Total Reward = {totalReward}")


torch.save(policyNN.state_dict(), "tictactoeM2.pth")
print("wins: ", wins)

for test in range(3):

    done = False
    state = env.reset(0)

    while not done:
        stateTensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = int(torch.argmax(policyNN(stateTensor)).item())

        state, nextState, reward, done = env.play(action, 1)

        env.print()
        if not done:
            action = int(input("enter your action: "))
            state, nextState, reward, done = env.play(action, 2)
            state = copy.deepcopy(nextState)
