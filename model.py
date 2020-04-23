import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """ This function is used for weight initialization. """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNet(nn.Module):
    """ Actor(Policy) Model. """

    def __init__(self, state_size, action_size, seed):
        """ 
        Initialize params and build model.

        params:
            - state_size (int)  : dimension of each state.
            - action_size (int) : dimension of each action.
            - seed (int)        : random seed.
        """
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input layer
        self.fc1 = nn.Linear(state_size, 64)
        # hidden layer
        self.fc2 = nn.Linear(64, 128)
        # hidden layer
        self.fc3 = nn.Linear(128, 64)
        # output layer
        self.fc4 = nn.Linear(64, action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        """ Builds a Actor network which maps states to actions. """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.tanh(self.fc4(x))


class CriticNet(nn.Module):
    """ Critic(Value) Model. """

    def __init__(self, state_size, action_size, seed):
        """ 
        Initialize params and build model.

        params:
            - state_size (int)  : dimension of each state.
            - action_size (int) : dimension of each action.
            - seed (int)        : random seed.
        """
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        # input layer
        self.fc1 = nn.Linear(state_size, 64)
        # hidden layer
        self.fc2 = nn.Linear(64, 128)
        # hidden layer
        self.fc3 = nn.Linear(128+action_size, 64)
        # output layer
        self.fc4 = nn.Linear(64, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        """ Builds a Critic network which maps (state, action) pairs to Qvalues. """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc3(x))

        return F.tanh(self.fc4(x))


