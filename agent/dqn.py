"""
Deep Q-Network implementation for SlytherNN.

Enhanced DQN with configurable architecture, dropout, and modern practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Action mappings - global for consistency
ACTIONS = [
    (0, -1),  # Up
    (0, 1),   # Down  
    (-1, 0),  # Left
    (1, 0),   # Right
]


class DQN(nn.Module):
    """
    Deep Q-Network with configurable architecture.
    
    Features:
    - Configurable hidden layer sizes
    - Dropout for regularization
    - Batch normalization option
    - Xavier initialization
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = None, dropout: float = 0.0,
                 use_batch_norm: bool = False):
        """
        Initialize DQN.
        
        Args:
            input_dim: Size of input state vector
            output_dim: Number of actions
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability (0 = no dropout)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]  # Default larger architecture
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout (except for last hidden layer)
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Single state tensor of shape (input_dim,)
            epsilon: Exploration probability
            
        Returns:
            Action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.output_dim, (1,)).item()
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for a state.
        
        Args:
            state: State tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Q-values tensor
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            return self.forward(state)
    
    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath: str, device: torch.device = None):
        """Load model state dict."""
        if device is None:
            device = next(self.parameters()).device
        
        state_dict = torch.load(filepath, map_location=device)
        self.load_state_dict(state_dict)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'architecture': str(self.network),
            'dropout': self.dropout,
            'batch_norm': self.use_batch_norm,
        }


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage estimation.
    
    This can provide better performance for some environments by explicitly
    modeling the state value and action advantages separately.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = None, dropout: float = 0.0):
        """
        Initialize Dueling DQN.
        
        Args:
            input_dim: Size of input state vector
            output_dim: Number of actions
            hidden_dims: List of hidden layer sizes for shared layers
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Shared feature layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Value stream (estimates V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream (estimates A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Dueling DQN.
        
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        """
        features = self.shared_layers(x)
        
        values = self.value_stream(features)  # Shape: (batch, 1)
        advantages = self.advantage_stream(features)  # Shape: (batch, num_actions)
        
        # Combine value and advantage streams
        # Subtract mean advantage for identifiability
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


def create_dqn_model(model_type: str = "dqn", **kwargs) -> nn.Module:
    """
    Factory function to create DQN models.
    
    Args:
        model_type: Type of model ("dqn" or "dueling")
        **kwargs: Model parameters
        
    Returns:
        DQN model instance
    """
    if model_type.lower() == "dqn":
        return DQN(**kwargs)
    elif model_type.lower() == "dueling":
        return DuelingDQN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Action utility functions
def action_to_direction(action_idx: int) -> tuple:
    """Convert action index to direction tuple."""
    if not 0 <= action_idx < len(ACTIONS):
        raise ValueError(f"Invalid action index: {action_idx}")
    return ACTIONS[action_idx]


def direction_to_action(direction: tuple) -> int:
    """Convert direction tuple to action index."""
    try:
        return ACTIONS.index(direction)
    except ValueError:
        raise ValueError(f"Invalid direction: {direction}")


def get_valid_actions(current_direction: tuple) -> list:
    """
    Get valid actions (cannot reverse direction).
    
    Args:
        current_direction: Current snake direction
        
    Returns:
        List of valid action indices
    """
    valid_actions = []
    reverse_direction = (-current_direction[0], -current_direction[1])
    
    for i, action_dir in enumerate(ACTIONS):
        if action_dir != reverse_direction:
            valid_actions.append(i)
    
    return valid_actions