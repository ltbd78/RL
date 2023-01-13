import numpy as np


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, maxlen):
        self.maxlen = maxlen
        self.pointer = 0
        self.size = 0

        self.state = np.zeros((maxlen, state_dim))
        self.action = np.zeros((maxlen, action_dim))
        self.reward = np.zeros((maxlen, 1))
        self.next_state = np.zeros((maxlen, state_dim))
        self.done = np.zeros((maxlen, 1))

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        self.state[self.pointer] = state
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.next_state[self.pointer] = next_state
        self.done[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def sample(self, batch_size):
        ind = np.random.randint(low=0, high=self.size, size=batch_size)
        return (self.state[ind], self.action[ind], self.reward[ind], self.next_state[ind], self.done[ind])
