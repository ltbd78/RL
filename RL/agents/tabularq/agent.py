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
        
        self.numbins = params['numbins']
        self.split = params['split'] # integer between 0 - 100
        self.resample = params['resample']
        self.resample_e = params['resample_e']
        
        self._build_agent()
        
    def _build_agent(self):
        self.memory = []
        self.bins = []
        self.bin_idx = []
        self.optim_steps = 0
        self.Qmatrix = np.random.uniform(low=-2, high=0, size=([self.numbins] * len(self.observation_space.high) + [self.action_space.n]))

        idx = 0
        for low, high in zip(self.observation_space.low, self.observation_space.high):
            if (low < -1e10 or high > 1e10):
                dis = sample(self.env)
                low_split, high_split = 0 + self.split/2, 1 - self.split/2
                low_idx, high_idx = int(len(dis[idx]) * low_split), int(len(dis[idx]) * high_split)
                lo, hi = dis[idx][low_idx], dis[idx][high_idx]
                self.bins.append(np.linspace(lo, hi, self.numbins))
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
        if (self.resample is True and self.optim_steps == self.resample_e):
            dis = self._resample_bins()
            print('Resampling...')
            
        # Monte Carlo Learning
        if self.target_update_freq is None: 
            for idx, (state, action, reward, next_state) in enumerate(self.memory):
                rewards = [t[2] for t in self.memory[idx:]]
                tdr = np.sum(get_total_discounted_rewards(rewards, self.gamma))
                self.Qmatrix[state + (action, )] += self.learning_rate * (reward + self.gamma * tdr)
                
        # Temporal Difference Learning
        else:  
            for state, action, reward, next_state in self.memory:
                self.Qmatrix[state + (action, )] += self.learning_rate * (reward + self.gamma * np.max(self.Qmatrix[next_state]) - self.Qmatrix[state + (action, )])
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.optim_steps += 1
        self.memory = []
        
    def get_discrete_state(self, state, bins, observation_size):
        stateIndex = []
        for i in range(observation_size):
            stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
        return tuple(stateIndex)

    def _explore_bins(self):
        pass
    
    def _resample_bins(self, epochs=1000):
        self.resample = False
        temp_epsilon = self.epsilon
        self.epislon = 0
        env = gym.make(self.env)
        env._max_episode_steps = 50000
        observation_space = env.observation_space
        action_space = env.action_space

        distributions = []
        for e in range(epochs):
            done = False
            x = []
            state = env.reset()
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                x.append(next_state)
                state = next_state
            distributions.append(np.mean(x, axis=0))
            # self.learn()

        res = []
        for i in range(len(distributions[0])):
            temp = [item[i] for item in distributions]
            temp.sort()
            res.append(temp)
            
        for idx in self.bin_idx:
                low_split, high_split = 0 + self.split/2, 1 - self.split/2
                low_split, high_split = int(len(res[0]) * low_split), int(len(res[0]) * high_split)
                lo, hi = res[idx][low_split], res[idx][high_split]
                self.bins[idx] = np.linspace(lo, hi, self.numbins)
        self.epsilon = temp_epsilon
    
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