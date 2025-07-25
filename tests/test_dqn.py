import torch
from agent.dqn import DQN
from snake_game.vector_env import VectorEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def test_dqn_forward_shape(): 
    """Test DQN forward pass with correct batch and state shape."""
    from snake_game.vector_env import VectorEnv
    envs = VectorEnv(num_envs=1, grid_size=12)
    state_dim = envs.get_states().shape[1]
    model = DQN(input_dim=state_dim, output_dim=4).to(device)
    dummy_input = torch.zeros((8, state_dim)).to(device)  # Batch of 8
    output = model(dummy_input)
    assert output.shape == (8, 4)
