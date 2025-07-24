import torch
import numpy as np
from agent.memory import ReplayMemory
from train import optimize_model
def test_optimize_model_returns_scalar_loss():
    model = DQN(144, 4)
    optimizer = torch.optim.Adam(model.parameters())
    memory = ReplayMemory(100)
    # Populate memory with a mock batch
    for _ in range(32):
        state = np.zeros((144,), dtype=np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.zeros((144,), dtype=np.float32)
        done = np.random.choice([True, False])
        memory.add((state, action, reward, next_state, done))
    loss = optimize_model(model, memory, optimizer, 32)
    assert isinstance(loss, float)
    assert loss >= 0

def test_optimize_model_changes_weights():
    model = DQN(144, 4)
    optimizer = torch.optim.Adam(model.parameters())
    memory = ReplayMemory(100)
    # Populate memory with a mock batch
    for _ in range(32):
        state = np.zeros((144,), dtype=np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.zeros((144,), dtype=np.float32)
        done = np.random.choice([True, False])
        memory.add((state, action, reward, next_state, done))
    # Save initial weights
    initial_weights = [p.clone() for p in model.parameters()]
    optimize_model(model, memory, optimizer, 32)
    # Check if any weights changed
    changed = any(not torch.equal(w, p) for w, p in zip(initial_weights, model.parameters()))
    assert changed
import pytest
from snake_game.game import SnakeGame
from agent.dqn import DQN
from train import select_action, run_episode

# Action space as defined in your train.py
ACTIONS = [
    (0, -1),  # Up
    (0, 1),   # Down
    (-1, 0),  # Left
    (1, 0),   # Right
]

def test_select_action_random_and_greedy():
    env = SnakeGame()
    model = DQN(input_dim=env.grid_size * env.grid_size, output_dim=4)
    state = env.get_state()

    # With epsilon=1 (full random), check action validity
    action = select_action(model, state, epsilon=1.0)
    assert isinstance(action, int)
    assert 0 <= action < len(ACTIONS)

    # With epsilon=0 (full greedy), check action validity
    action = select_action(model, state, epsilon=0.0)
    assert isinstance(action, int)
    assert 0 <= action < len(ACTIONS)

def test_run_episode_basic():
    env = SnakeGame()
    model = DQN(input_dim=env.grid_size * env.grid_size, output_dim=4)
    from agent.memory import ReplayMemory
    import torch
    memory = ReplayMemory(100)
    optimizer = torch.optim.Adam(model.parameters())
    total_reward, steps, _ = run_episode(env, model, epsilon=0.5, memory=memory, optimizer=optimizer)
    assert isinstance(total_reward, (int, float))
    assert isinstance(steps, int)
    assert 0 < steps <= 250  # default max steps per episode

def test_epsilon_decay():
    from train import EPS_START, EPS_END, EPS_DECAY
    epsilon = EPS_START
    for _ in range(100):
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        assert EPS_END <= epsilon <= EPS_START

if __name__ == "__main__":
    pytest.main()
