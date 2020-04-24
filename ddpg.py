import numpy as np
import random
import copy
from collections import namedtuple, deque

from oldmodel import ActorNet, CriticNet

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update hyperparameter
LR_ACTOR = 1e-3         # actor learning rate
LR_CRITIC = 1e-3        # critic learning rate
LEARN_EVERY = 20        # lerning every 
LEARN_FOR = 10          # learn for these many intervals
EPSILON_START = 1.0     # starting value for eps
EPSILON_END = 0.05      # minimum value for eps
EPSILON_DECAY = 1e-6    # decay rate for eps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """ A DDPG Agent which interacts and learns from the environment. """

    def __init__(self, state_size, action_size, seed):
        """
        
        Initializes a DDPG Agent.

        params:
            - state_size (int)  : dimension of each state.
            - action_size (int) : dimension of each action.
            - seed (int)        : random seed.

        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.eps = EPSILON_START

        # Setup Actor Network
        self.actor_net = ActorNet(self.state_size, self.action_size, seed).to(device)
        self.target_actor_net = ActorNet(self.state_size, self.action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)

        # Setup Critic Network
        self.critic_net = CriticNet(self.state_size, self.action_size, seed).to(device)
        self.target_critc_net = CriticNet(self.state_size, self.action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR_CRITIC)

        # noise process
        self.noise = OUNoise(self.action_size, seed)

        # create replay buffer
        self.buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # timestep counter
        self.tstep = 0

    
    def step(self, states, actions, rewards, next_states, dones):
        # iterate through 20 agents
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # save experiences in replay buffer
            self.buffer.push(state, action, reward, next_state, done)

        # Learn every C timesteps
        self.tstep = (self.tstep+1) % LEARN_EVERY

        if self.tstep == 0:
            # check if enough samples are available in buffer
            if len(self.buffer) > BATCH_SIZE:
                # Learn for a few iterations
                for _ in range(LEARN_FOR):
                    experiences = self.buffer.sample()
                    self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):
        """ 
        Updates policy and value params using given batch of experience tuples. 
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) = action
            critic_target(state, action) = Qvalue

        params:
            - experiences (Tuple([torch.Tensor])) : tuple of (s, a, r, s', done).
            - gamma (float)                       : discount factor.        
        """

        # unpack experiences
        s, a, r, ns, d = experiences

        #################### Update Critic ####################
        # get predicted next state actions from target models
        next_actions = self.target_actor_net(ns)
        # get predicted next state and Q values from target models
        next_Q_targets = self.target_critc_net(ns, next_actions)

        # Compute Q targets for current states 
        Q_targets = r + (gamma * next_Q_targets * (1 - d))

        # Compute critic loss
        Q_expected = self.critic_net(s, a)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm(self.critic_net.parameters(), 1.0)
        self.critic_optimizer.step()

        #######################################################

        #################### Update Actor ####################

        # compute actor loss
        predicted_actions = self.actor_net(s)
        actor_loss = - self.critic_net(s, predicted_actions).mean()

        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #######################################################

        #################### Update Target Networks ####################
        self.soft_update(self.critic_net, self.target_critc_net, TAU)
        self.soft_update(self.actor_net, self.target_actor_net, TAU)

        # decay epsilon
        if self.eps > EPSILON_END:
            self.eps *= EPSILON_DECAY
            self.noise.reset()
        else:
            self.eps = EPSILON_END
    
    def soft_update(self, local, target, tau):
        """
        Performs a soft update for the parameters.
        theta_target = tau * theta_local + (1 - tau) * theta_target
        
        params:
            - TAU (float) : interpolation parameter. 
        """

        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    
    def reset(self):
        """ This function resets the noise. """
        self.noise.reset()
    
    def act(self, state, add_noise=True):
        """ 
        Returns actions for a given state as per current policy.

        params:
            - state (array like)  : current state.
            - add_noise (boolean) : flag for adding noise.
        """

        state = torch.from_numpy(state).float().to(device)

        # set actor to eval mode
        self.actor_net.eval()

        with torch.no_grad():
            # get action values
            act_vals = self.actor_net(state).cpu().data.numpy()

        # turn back to train mode
        self.actor_net.train()

        # add noise
        if add_noise:
            act_vals += self.noise.sample()*self.eps

        

        return np.clip(act_vals, -1, 1)        







class OUNoise:
    """ Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """ Initializes parameters and the noise process. """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """ Resets the internal state (=noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """ Updates internal state and returns it as a noise sample. """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state

class ReplayBuffer:
    """ Replay Buffer which stores experience tuples. """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ 
        Initializes a RB object.

        params:
            - action_size (int) : dimension of each action.
            - buffer_size (int) : max size of buffer.
            - batch_size (int)  : size of each training batch.
            - seed (int)        : random seed.
        
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        # creates the replay buffer
        self.buffer = deque(maxlen=buffer_size)
        # creates a namedtuple for experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def push(self, state, action, reward, next_state, done):
        """
        Adds new experience to the replay buffer.
        
        params:
            - state      : current state of the environment.
            - action     : action taken by the agent in the current state.
            - reward     : reward received for taking action in current state.
            - next_state : next state which the agent transitions to after taking action in the current state.
            - done       : flag which determines if the experience has ended.
        """
        # create namedtuple
        exp = self.experience(state, action, reward, next_state, done)
        # append to buffer
        self.buffer.append(exp)
    
    def sample(self):
        """
        Randomly samples a batch of experiences from the replay buffer.
        """

        # get experiences
        experiences = random.sample(self.buffer, k=self.batch_size)

        # stack the experiences into different torch tensors
        s_, a_, r_, ns_, d_ = [], [], [], [], []

        # use single loop instead of creating a generator for each
        for e in experiences:
            if e is not None:
                s_.append(e.state)
                a_.append(e.action)
                r_.append(e.reward)
                ns_.append(e.next_state)
                d_.append(e.done)
        
        states = torch.from_numpy(np.vstack(s_)).float().to(device)
        actions = torch.from_numpy(np.vstack(a_)).float().to(device)
        rewards = torch.from_numpy(np.vstack(r_)).float().to(device)
        next_states = torch.from_numpy(np.vstack(ns_)).float().to(device)
        dones = torch.from_numpy(np.vstack(d_).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.buffer)
        