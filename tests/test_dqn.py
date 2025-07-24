import torch
from agent.dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def test_dqn_forward_shape():
    model = DQN(input_dim=144, output_dim=4).to(device)  # Example: 12x12 grid flattened, 4 actions
    dummy_input = torch.zeros((2, 144)).to(device)  # Batch of 2
    output = model(dummy_input)
    assert output.shape == (2, 4)
