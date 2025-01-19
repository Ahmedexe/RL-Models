import copy
import numpy as np
from random import sample


class TicTacToe:
    def __init__(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def play(self, action, player):
        prevState = copy.deepcopy(self.state)
        if self.state[action] == 0:
            self.state[action] = player
        reward = rewardCalc(prevState, self.state, action)
        done = isDone(self.state)

        next = copy.deepcopy(self.state)

        return prevState, next, reward, done

    def reset(self, player):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.state[np.random.randint(8)] = float(sample([0, player], 1)[0])
        return self.state

    def print(self):
        for i in range(3):
            for j in range(3):
                print(self.state[j + 3*i])
            print()


def rewardCalc(prevState, nextState, action):

    if prevState[action] != 0:
        reward = -10
    elif isDone(nextState):
        reward = 10
    else:
        reward = -1

    return reward


def isDone(state):

    # check rows
    for i in range(3):
        if state[0 + 3*i] == state[1 + 3*i] == state[2 + 3*i] != 0:
            return 1

        if state[0 + i] == state[3 + i] == state[6 + i] != 0:
            return 1

    if state[0] == state[4] == state[8] != 0:
        return 1

    if state[2] == state[4] == state[6] != 0:
        return 1

    flag = 1
    for k in range(9):
        if (state[k] == 0):
            flag = 0

    return flag
