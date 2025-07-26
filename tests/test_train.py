import torch
import numpy as np
import pytest
from agent.dqn import DQN, ACTIONS
from agent.prioritized_memory import PrioritizedReplayMemory
from snake_game.vector_env import VectorEnv
from snake_game.game import SnakeGame
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

def test_simple_epsilon_decay():
    """Test the simple epsilon decay used in train.py."""
    from train import EPS_START, EPS_END, EPS_DECAY
    
    epsilon = EPS_START  # Should be 1.0
    assert epsilon == 1.0, f"Starting epsilon should be 1.0, got {epsilon}"
    
    # Test decay formula: epsilon = max(EPS_END, epsilon * EPS_DECAY)
    for _ in range(10):
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    assert epsilon <= EPS_START, "Epsilon should decrease from start value"
    assert epsilon >= EPS_END, "Epsilon should not go below EPS_END"
    
    # Test that it eventually reaches EPS_END
    epsilon = EPS_START
    for _ in range(1000):  # Many decay steps
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    assert epsilon == EPS_END, f"Epsilon should eventually reach EPS_END ({EPS_END}), got {epsilon}"

# ============= WIN CONDITION TRAINING TESTS =============

def test_win_vs_death_rewards_training():
    """Test that win gives positive reward while death gives negative in training context."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test win scenario
    win_game = SnakeGame(grid_size=3, reward_win=100, reward_death=-10, reward_fruit=5)
    # Create valid snake body covering 8/9 cells
    win_game.snake.body = [
        (0, 1),  # Head at middle-left
        (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)  # Body in spiral
    ]
    win_game.fruit.position = (1, 1)  # Center cell
    win_game.snake.direction = (1, 0)  # Moving right to fruit
    
    _, win_reward, win_done = win_game.step(3, device)  # Move right
    
    # Test death scenario
    death_game = SnakeGame(grid_size=3, reward_win=100, reward_death=-10, reward_fruit=5)
    death_game.snake.body = [(2, 2)]
    death_game.snake.direction = (1, 0)  # Move into wall
    
    _, death_reward, death_done = death_game.step(3, device)  # Move right into wall
    
    assert win_reward == 100, f"Win should give +100 reward, got {win_reward}"
    assert death_reward == -10, f"Death should give -10 reward, got {death_reward}"
    assert win_done and death_done, "Both scenarios should end the game"

def test_win_condition_in_vector_env():
    """Test that win condition works correctly in vectorized training environment."""
    envs = VectorEnv(num_envs=2, grid_size=3)
    
    # Set up env 0 to win
    envs.envs[0].snake.body = [
        (0, 1),  # Head at middle-left
        (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)  # Body
    ]
    envs.envs[0].fruit.position = (1, 1)  # Center cell
    envs.envs[0].snake.direction = (1, 0)  # Moving right
    
    # Set up env 1 to die
    envs.envs[1].snake.body = [(2, 2)]
    envs.envs[1].fruit.position = (0, 0)
    envs.envs[1].snake.direction = (1, 0)  # Will hit wall
    
    # Take actions: right for env0 (win), right for env1 (die)
    actions = torch.tensor([3, 3])  # right, right
    next_states, rewards, dones = envs.step(actions)
    
    assert rewards[0].item() == 100, f"Env 0 should win with +100 reward, got {rewards[0].item()}"
    assert rewards[1].item() == -10, f"Env 1 should die with -10 reward, got {rewards[1].item()}"
    assert dones[0].item() == True, "Env 0 should be done after winning"
    assert dones[1].item() == True, "Env 1 should be done after dying"

def test_reward_hierarchy_training():
    """Test that reward hierarchy is correct: Win > Fruit > Step > Death."""
    game = SnakeGame(grid_size=3, reward_win=100, reward_fruit=5, reward_step=-0.01, reward_death=-10)
    
    # Test reward values are in correct hierarchy
    assert game.reward_win > game.reward_fruit, "Win reward should be higher than fruit reward"
    assert game.reward_fruit > game.reward_step, "Fruit reward should be higher than step reward" 
    assert game.reward_step > game.reward_death, "Step reward should be higher than death penalty"
    
    # Test that win reward incentivizes perfect play
    max_possible_fruit_reward = (3 * 3 - 3) * game.reward_fruit  # 6 fruits max * 5 = 30
    assert game.reward_win > max_possible_fruit_reward, "Win reward should exceed sum of all possible fruit rewards"

def test_training_memory_with_win_rewards():
    """Test that win rewards are properly stored in training memory."""
    memory = PrioritizedReplayMemory(capacity=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy states
    state = torch.zeros(150, dtype=torch.float32)  # Assuming 12x12 + 4 + 2 = 150
    next_state = torch.zeros(150, dtype=torch.float32)
    
    # Add win experience
    memory.add((state, 0, 100.0, next_state, True))  # Win reward
    
    # Add death experience  
    memory.add((state, 1, -10.0, next_state, True))  # Death penalty
    
    # Add fruit experience
    memory.add((state, 2, 5.0, next_state, False))  # Fruit reward
    
    # Sample from memory
    batch, idxs, weights = memory.sample(3)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Check that all reward types are preserved
    reward_values = [r for r in rewards]
    assert 100.0 in reward_values, "Win reward should be in memory"
    assert -10.0 in reward_values, "Death penalty should be in memory"  
    assert 5.0 in reward_values, "Fruit reward should be in memory"

def test_training_hyperparameters_match():
    """Test that training hyperparameters are accessible and have expected values."""
    from train import EPS_START, EPS_END, EPS_DECAY, LEARNING_RATE, BATCH_SIZE, NUM_ENVS
    
    # Test epsilon parameters
    assert EPS_START == 1.0, f"EPS_START should be 1.0, got {EPS_START}"
    assert EPS_END == 0.05, f"EPS_END should be 0.05, got {EPS_END}"
    assert EPS_DECAY == 0.995, f"EPS_DECAY should be 0.995, got {EPS_DECAY}"
    
    # Test other hyperparameters are reasonable
    assert LEARNING_RATE > 0, "Learning rate should be positive"
    assert BATCH_SIZE > 0, "Batch size should be positive"
    assert NUM_ENVS > 0, "Number of environments should be positive"

def test_gradient_accumulation_steps():
    """Test that gradient accumulation is working as configured."""
    from train import GRAD_ACCUM_STEPS
    
    assert GRAD_ACCUM_STEPS >= 1, "Gradient accumulation steps should be at least 1"
    assert isinstance(GRAD_ACCUM_STEPS, int), "Gradient accumulation steps should be an integer"