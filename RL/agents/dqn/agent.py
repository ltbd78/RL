import numpy as np
import gym
import copy
import torch
import random

from RL.agents.dqn.network import *
from RL.agents.utils.replay_buffer import *


class DQNAgent:
    def __init__(self, observation_space, action_space, **params):
        # Environment Params
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(observation_space, gym.spaces.Box)
        self.state_dim = observation_space.shape[0]
        assert isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n
        
        # Agent Common Params (ordered by preference)
        self.online = params['online']
        self.gamma = params['gamma'] # Discount factor
        self.hidden_dim = params['hidden_dim']
        self.learning_rate = params['learning_rate']
        self.device = torch.device(params['device'])
        if params['dtype'] == 'float32':
            self.dtype = torch.float32
        if params['dtype'] == 'float64':
            self.dtype = torch.float64
        
        # Agent Specific Params (ordered alphabetically)
        self.batch_size = params['batch_size'] # size to sample from memory
        self.clip = params['clip']
        self.dueling = params['dueling']
        self.epsilon = params['epsilon'] # initial exploration rate
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.memory_maxlen = params['memory_maxlen']
        self.per = params['per']
        assert not (self.per == True and self.online == False)
        if self.per:
            self.memory_alpha = params['memory_alpha']
            self.memory_beta = params['memory_beta'] # initial beta; approaches 1 as epsilon approaches 0
        self.target_update_freq = params['target_update_freq'] # double network

        self._build_agent()
  
    def _build_agent(self):
        # Networks
        self.optim_steps = 0
        self.network = DQN(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, dueling=self.dueling).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)
        
        # Memory
        if self.per:
            self.memory = PrioritizedReplayBuffer(size=self.memory_maxlen, alpha=self.memory_alpha)
        else:
            self.memory = ReplayBuffer(size=self.memory_maxlen)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def get_action(self, state):
        if random.random() <= self.epsilon:
            action = int(random.random()*self.action_dim) # random action
        else:
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
            act_values = self.network(state)
            action = np.argmax(act_values.data.cpu().numpy()) # predicted action
        return action
    
    def learn(self):
        n = self.batch_size
        if self.per:
            minibatch = self.memory.sample(batch_size=n, beta=max(self.memory_beta, 1-self.epsilon))
            weights = torch.tensor(minibatch[5], dtype=self.dtype, device=self.device).reshape(-1, 1)
            idx = minibatch[6]
        else:
            minibatch = self.memory.sample(batch_size=n)
        s0 = torch.tensor(minibatch[0], dtype=self.dtype, device=self.device)
        a0 = torch.tensor(minibatch[1], dtype=torch.int64, device=self.device).reshape(-1, 1)
        r = torch.tensor(minibatch[2], dtype=self.dtype, device=self.device).reshape(-1, 1)
        s1 = torch.tensor(minibatch[3], dtype=self.dtype, device=self.device)
        d = torch.tensor(minibatch[4], dtype=torch.int64, device=self.device).reshape(-1, 1)

        # let subscripts 0 := current and 1 := next
        # let Q' be the double network ("target_network") that lags behind Q
        Q_s0_a0 = torch.gather(self.network(s0), 1, a0) # Q(s=s0, a=a0)
        Q_s1 = self.network(s1) # Q(s=s1, a=.)
        a1 = Q_s1.argmax(dim=1).reshape(-1, 1) # a1 = argmax_a Q(s=s1, a=.) ; `a1` is always chosen by original Q
        if self.target_update_freq is not None:
            if self.optim_steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            Q_s1 = self.target_network(s1) # Q'(s=s1, a=.) ; if using double, Q' will be used to eval `a1` chosen by original Q 
        
        Q_expected = r + self.gamma*(1-d)*torch.gather(Q_s1, 1, a1) # Q_expected = r + gamma*(1-d)*{Q or Q'}(s=s1, a=a1)
        
        errors = (Q_expected - Q_s0_a0)
        
        if self.per: # TODO: results show it is not working as intended
            priorities = np.abs(errors.data.cpu().numpy()) + 1e-3 # check dtype
            self.memory.update_priorities(idx, priorities)
            loss = (errors.pow(2)*weights).mean()
        else:
            loss = errors.pow(2).mean()
        
        self.optim.zero_grad()
        loss.backward()
        
        if self.clip:
            for param in self.network.parameters():
                param.grad.data.clamp_(-1, 1)
                
        self.optim.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        
        self.optim_steps += 1

    def save(self, path):
        torch.save({'network': self.network.state_dict(),
                    'optim': self.optim.state_dict()},
                   path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)
        
        self.epsilon = .01 # for testing