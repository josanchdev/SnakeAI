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

    total_reward, steps = run_episode(env, model, epsilon=0.5)
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
