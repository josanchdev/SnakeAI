import torch
import torch.nn as nn        # Neural network layers and operations
import torch.optim as optim  # Optimization algorithms (e.g., Adam, SGD)
import torch.nn.functional as F  # Functional API for layers (e.g., activation functions)

class DQN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(DQN, self).__init__()
		# Larger architecture for full GPU utilization
		self.fc1 = nn.Linear(input_dim, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 512)
		self.fc4 = nn.Linear(512, 256)
		self.fc5 = nn.Linear(256, output_dim)

	def forward(self, x):
		# Forward pass through the larger network
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = self.fc5(x)
		return x