import torch
import torch.nn as nn        # Neural network layers and operations
import torch.optim as optim  # Optimization algorithms (e.g., Adam, SGD)
import torch.nn.functional as F  # Functional API for layers (e.g., activation functions)

class DQN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(DQN, self).__init__()
		# Define the neural network architecture
		self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
		self.fc2 = nn.Linear(128, 128)         # Second fully connected layer
		self.fc3 = nn.Linear(128, output_dim)  # Output layer

	def forward(self, x):
		# Forward pass through the network
		x = F.relu(self.fc1(x))  # Apply ReLU activation after first layer
		x = F.relu(self.fc2(x))  # Apply ReLU activation after second layer
		x = self.fc3(x)           # Output layer (no activation function)
		return x