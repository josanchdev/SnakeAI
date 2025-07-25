import torch
import numpy as np
import pytest
from agent.dqn import DQN, ACTIONS
from agent.prioritized_memory import PrioritizedReplayMemory
from snake_game.vector_env import VectorEnv
from train import select_actions_batch, optimize_model
from train import select_actions_batch, optimize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_dqn_forward_shape():
    """Test DQN forward pass with correct batch and state shape."""
    envs = VectorEnv(num_envs=1, grid_size=12)
    state_dim = envs.get_states().shape[1]
    model = DQN(input_dim=state_dim, output_dim=4).to(device)
    dummy_input = torch.zeros((8, state_dim)).to(device)
    output = model(dummy_input)
    assert output.shape == (8, 4)

def test_select_actions_batch_random_and_greedy():
    """Test batched epsilon-greedy action selection."""
    envs = VectorEnv(num_envs=1, grid_size=12)
    state_dim = envs.get_states().shape[1]
    model = DQN(state_dim, 4).to(device)
    dummy_states = torch.zeros((16, state_dim)).to(device)
    # Epsilon = 1.0: all random
    actions = select_actions_batch(model, dummy_states, epsilon=1.0)
    assert actions.shape == (16,)
    assert ((0 <= actions) & (actions < 4)).all()
    # Epsilon = 0.0: all greedy
    actions2 = select_actions_batch(model, dummy_states, epsilon=0.0)
    assert ((0 <= actions2) & (actions2 < 4)).all()

def test_vector_env_step_and_reset():
    """Test VectorEnv batch step, reset, and output shapes."""
    envs = VectorEnv(num_envs=4, grid_size=6)
    states = envs.reset()
    assert states.shape[0] == 4  # 4 environments
    actions = torch.zeros(4, dtype=torch.long)
    next_states, rewards, dones = envs.step(actions)
    assert next_states.shape[0] == 4
    assert rewards.shape[0] == 4
    assert dones.shape[0] == 4

def test_optimize_model_with_per():
    """Test optimize_model with PrioritizedReplayMemory and correct state shapes."""
    envs = VectorEnv(num_envs=1, grid_size=12)
    state_dim = envs.get_states().shape[1]
    model = DQN(state_dim, 4).to(device)
    # Use the same optimizer as in train.py
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters())
    memory = PrioritizedReplayMemory(capacity=32)
    # Fill memory with matching-sized states
    for _ in range(32):
        state = torch.zeros((state_dim,), dtype=torch.float32)
        action = 0
        reward = 1.0
        next_state = torch.zeros((state_dim,), dtype=torch.float32)
        done = False
        memory.add((state, action, reward, next_state, done))
    # Patch global memory in optimize_model
    import train
    train.memory = memory
    loss = optimize_model()
    assert loss is None or isinstance(loss, float)

import torch

def test_steps_since_last_reward_logic():
    """
    Test the 'steps_since_last_reward' counter logic that:
    - resets to zero only when reward strictly increases (fruit eaten),
    - increments otherwise,
    - triggers done when threshold is reached.

    Uses a small threshold for quick testing.
    """

    NUM_ENVS = 3
    MAX_STEPS_SINCE_REWARD = 5  # small for test speed
    steps_since_last_reward = torch.zeros(NUM_ENVS)
    last_rewards = torch.zeros(NUM_ENVS)
    dones = torch.zeros(NUM_ENVS, dtype=torch.bool)

    # Simulate 10 steps of reward sequences per env:
    # Env 0 never gets reward (should be done at step 5)
    # Env 1 gets reward at step 2 and never again (should be done at step 7)
    # Env 2 gets rewards at steps 4 and 7 (should NOT be done)
    rewards_seq = [
        torch.tensor([0.0, 0.0, 0.0]),  # step 1
        torch.tensor([0.0, 1.0, 0.0]),  # step 2 (env1 gets fruit)
        torch.tensor([0.0, 1.0, 0.0]),  # step 3
        torch.tensor([0.0, 1.0, 1.0]),  # step 4 (env2 gets fruit)
        torch.tensor([0.0, 1.0, 1.0]),  # step 5
        torch.tensor([0.0, 1.0, 1.0]),  # step 6
        torch.tensor([0.0, 1.0, 2.0]),  # step 7 (env2 gets fruit again)
        torch.tensor([0.0, 1.0, 2.0]),  # step 8
        torch.tensor([0.0, 1.0, 2.0]),  # step 9
        torch.tensor([0.0, 1.0, 2.0]),  # step 10
    ]

    for t, rewards in enumerate(rewards_seq):
        for i in range(NUM_ENVS):
            # Reset steps counter only if reward strictly increased (fruit eaten)
            if rewards[i].item() > last_rewards[i].item():
                steps_since_last_reward[i] = 0
            else:
                steps_since_last_reward[i] += 1

            last_rewards[i] = rewards[i]

            # Trigger done if threshold reached
            if steps_since_last_reward[i] >= MAX_STEPS_SINCE_REWARD:
                dones[i] = True

    # Assertions:
    # Env 0 never got fruit → done after 5 steps
    assert dones[0].item() == 1, "Env 0 should be done after 5 steps without reward"

    # Env 1 got fruit at step 2 → done after 5 steps since then → step 7 (counting from step 3)
    assert dones[1].item() == 1, "Env 1 should be done after 5 steps following last fruit at t=2"

    # Env 2 got fruit twice (step 4 and 7) so counter reset twice: should NOT be done
    assert dones[2].item() == 0, "Env 2 should NOT be done due to resetting steps at fruit"


