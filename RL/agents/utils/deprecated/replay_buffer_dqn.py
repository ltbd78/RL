import random
import numpy as np
from collections import deque

# Inspiration from TheComputerScientist @ Youtube

# TODO: fix PER results
# resource: https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb

class ReplayMemory(): # TODO: swap deque with binary heap / sum tree
    def __init__(self, maxlen, per=False):
        self.maxlen = maxlen
        self.per = per
        self.buffer = deque(maxlen=self.maxlen)
        if self.per:
            self.priorities = deque(maxlen=self.maxlen)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)
        if self.per:
            self.priorities.append(max(self.priorities, default=1))

    def _get_probabilities(self, damping):
        scaled_priorities = np.array(self.priorities) ** damping
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def _get_importances(self, probabilities):
        importances = 1/len(self.buffer) * 1/probabilities
        importances_normalized = importances / max(importances)
        return importances_normalized

    def sample(self, batch_size, damping=1.0):
        if self.per:
            sample_size = min(len(self.buffer), batch_size)
            sample_probs = self._get_probabilities(damping)
            indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs) # TODO: test performance with numpy instead of random
            samples = np.array(self.buffer)[indices]
            importances = self._get_importances(sample_probs[indices])
            return samples, importances, indices
        else:
            return random.sample(self.buffer, batch_size)
