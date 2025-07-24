from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} experiences from memory of size {len(self.memory)}")
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def is_full(self):
        return len(self.memory) == self.max_size