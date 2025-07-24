import torch
from agent.dqn import DQN

def test_dqn_forward_shape():
    model = DQN(input_dim=144, output_dim=4)  # Example: 12x12 grid flattened, 4 actions
    dummy_input = torch.zeros((2, 144))  # Batch of 2
    output = model(dummy_input)
    assert output.shape == (2, 4)
