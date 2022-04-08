import numpy as np
import gym
import random
import pickle

from RL.agents.utils.misc import *
from RL.agents.random.agent import *


class TabularQ:
    def __init__(self, observation_space, action_space, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        self.online = params['online']
        self.target_update_freq = params['target_update_freq']
        self.env = params['environment']
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.gamma = params['gamma']
        self.learning_rate = params['learning_rate']
        self.lr_decay = params['lr_decay']
        self.lr_min = params['lr_min']
        
        self.bin_range = params['bin_range'] # list of tuples (high, low), or None
        self.numbins = params['numbins']
        self.split = params['split'] # integer between 0 - 100, or None 
        
        self._build_agent()
        
    def _build_agent(self):
        self.memory = []
        self.bins = []
        self.bin_idx = []
        self.optim_steps = 0
        self.Qmatrix = np.random.uniform(low=-2, high=0, size=([self.numbins] * len(self.observation_space.high) + [self.action_space.n]))
        
        # user specified bins
        if self.bin_range is not None:
            for low, high in self.bin_range:
                self.bins.append(np.linspace(low, high, self.numbins))
        
        # auto-generated bins
        if self.split is not None:
            idx = 0
            for low, high in zip(self.observation_space.low, self.observation_space.high):
                if (low < -1e10 or high > 1e10):
                    state_distribution = sample(self.env) # mes (max episode steps), optional parameter epochs
                    if self.split != 0:
                        left, right = 0 + self.split/2, 1 - self.split/2
                        low_idx, high_idx = int(len(dis[idx]) * left), int(len(dis[idx]) * right)
                        lower_bound, upper_bound = state_distribution[idx][low_idx], state_distribution[idx][high_idx]
                    else:
                        lower_bound, upper_bound = min(state_distribution[idx]), max(state_distribution[idx])
            
                    self.bins.append(np.linspace(lower_bound, upper_bound, self.numbins))
                    self.bin_idx.append(idx)
                else:
                    self.bins.append(np.linspace(low, high, self.numbins))
                idx += 1
        
    def get_action(self, state):
        state = self.get_discrete_state(state, self.bins, len(self.observation_space.high))
        if (self.epsilon > np.random.random()):
            return self.action_space.sample()
        else:
            return np.argmax(self.Qmatrix[state])
        
    def remember(self, state, action, reward, next_state, done):
        state = self.get_discrete_state(state, self.bins, len(self.observation_space.high))
        next_state = self.get_discrete_state(next_state, self.bins, len(self.observation_space.high))
        self.memory.append((state, action, reward, next_state))
        
    def learn(self):
        """
        Monte Carlo Learning:
        Q(s_t, a_t) = Q(s_t, a_t) + α * [G_t - Q(s_t, a_t)]
            where:
            G_t = R_t+1 + γ * R_t+2 + ... + γ^T-1 * R_T (total discounted rewards)
            T = termination timestep 
        
        Temporal Difference Learning:
        Q(s_t, a_t) = Q(s_t, a_t) + α * [R_(t+1) + γ * Q(s_t+1, a) - Q(s_t, a_t)]
            where:
            R_(t+1) = reward at given time step
            Q(s_t+1, a) = optimal value function given the next state
        """
        if self.target_update_freq is None: 
            for idx, (state, action, reward, next_state) in enumerate(self.memory):
                rewards = [t[2] for t in self.memory[idx:]]
                tdr = np.sum(get_total_discounted_rewards(rewards, self.gamma))
                self.Qmatrix[state + (action, )] += self.learning_rate * (tdr - self.Qmatrix[state + (action, )])      
        else:  
            for state, action, reward, next_state in self.memory:
                self.Qmatrix[state + (action, )] += self.learning_rate * (reward + self.gamma * np.max(self.Qmatrix[next_state]) - self.Qmatrix[state + (action, )])
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if (self.lr_min and self.lr_decay) is not None:
            self.learning_rate = max(self.lr_min, self.lr_decay * self.learning_rate)
            
        self.optim_steps += 1
        self.memory = []
        
    def get_discrete_state(self, state, bins, observation_size):
        state_idx = []
        for i in range(observation_size):
            state_idx.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
        return tuple(state_idx)
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'Qmatrix': self.Qmatrix, 'bins': self.bins, 'bin_idx': self.bin_idx}, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            file = pickle.load(f)
            self.Qmatrix = file['Qmatrix']
            self.bins = file['bins']
            self.bin_idx = file['bin_idx']
            
        self.epsilon = 0.01 # for testing