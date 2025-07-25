import torch
import pytest
from snake_game.vector_env import VectorEnv

def test_vector_env_batch_shapes():
    """Test that VectorEnv returns correct batch shapes for states, rewards, and dones."""
    envs = VectorEnv(num_envs=8, grid_size=8)
    states = envs.reset()
    assert states.shape[0] == 8
    actions = torch.zeros(8, dtype=torch.long)
    next_states, rewards, dones = envs.step(actions)
    assert next_states.shape[0] == 8
    assert rewards.shape[0] == 8
    assert dones.shape[0] == 8

def test_vector_env_auto_reset():
    """Test that VectorEnv auto-resets done environments."""
    envs = VectorEnv(num_envs=2, grid_size=5)
    envs.reset()
    # Force one env to done
    envs.envs[0].running = False
    actions = torch.zeros(2, dtype=torch.long)
    # Should auto-reset env 0
    _, _, _ = envs.step(actions)
    assert envs.envs[0].running
