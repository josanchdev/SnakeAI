import numpy as np
import random

class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    Used for efficient sampling and updating priorities in PER.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    @property
    def total(self):
        return self.tree[0]

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-4, eps=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps
        self.max_priority = 1.0

    def add(self, transition):
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_probabilities = np.array(priorities) / self.tree.total
        is_weights = np.power(self.tree.n_entries * sampling_probabilities + self.eps, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, np.abs(priority) + self.eps)
            self.max_priority = max(self.max_priority, np.abs(priority) + self.eps)

    def __len__(self):
        return self.tree.n_entries
