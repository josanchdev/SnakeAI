import torch
import torch.nn as nn        # Neural network layers and operations
import torch.optim as optim  # Optimization algorithms (e.g., Adam, SGD)
import torch.nn.functional as F  # Functional API for layers (e.g., activation functions)

"""
DQN agent and RL constants.
"""

# Global action list for all modules
ACTIONS = [
	(0, -1),  # Up
	(0, 1),   # Down
	(-1, 0),  # Left
	(1, 0),   # Right
]

class DQN(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim=128):
		"""
		Deep Q-Network with configurable hidden size.
		Args:
			input_dim (int): State vector size
			output_dim (int): Number of actions
			hidden_dim (int): Hidden layer size
		"""
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		"""
		Forward pass through the DQN.
		Args:
			x (torch.Tensor): Input state(s), shape (batch, input_dim)
		Returns:
			torch.Tensor: Q-values for each action
		"""
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x