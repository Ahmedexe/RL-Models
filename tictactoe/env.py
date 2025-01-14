import copy


class TicTacToe:
    def __init__(self, player):
        self.state = ["0","0","0","0","0","0","0","0","0"]
        self.player = player

    def play(self, state, action):
        prevState = copy.deepcopy(state)
        state[action] = self.player
        reward = rewardCalc(prevState, state, action)
        done = isWon(state)

        return state, reward, done
    
    def reset(self):
        self.state = ["0","0","0","0","0","0","0","0","0"]
        return self.state






def rewardCalc(prevState, nextState, action):

    if prevState[action] != "0":
        reward = -10
    elif isWon(nextState):
        reward = 10 
    else:
        reward = 1

    return reward

def isWon(state):

    # check rows
    for i in range(3):
        if state[0 + 3*i] == state[1 + 3*i] == state[2 + 3*i] != "0":
            return 1
        
        if state[0 + i] == state[3 + i] == state[6 + i] != "0":
            return 1
        
    if state[0] == state[4] == state[8] != "0":
        return 1
    
    if state[2] == state[4] == state[6] != "0": 
        return 1

    return 0
 