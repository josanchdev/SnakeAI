# train.py
import torch
import numpy as np
from agent.dqn import DQN
from agent.memory import ReplayMemory
from snake_game.game import SnakeGame

# ==== Hyperparameters ====
NUM_EPISODES = 20          # Start small for debugging
MAX_STEPS_PER_EP = 250     # To prevent runaway loops
MEMORY_SIZE = 1000
BATCH_SIZE = 32
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995  # Decay epsilon by this factor per episode
LEARNING_RATE = 1e-3

# ==== Helper: Map actions ====
ACTIONS = [
    (0, -1),  # Up
    (0, 1),   # Down
    (-1, 0),  # Left
    (1, 0),   # Right
]

def select_action(model, state, epsilon):
    # Choose random or best action
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def run_episode(env, model, epsilon):
    state = env.get_state()
    total_reward = 0
    env.running = True
    steps = 0

    while env.running and steps < MAX_STEPS_PER_EP:
        action_idx = select_action(model, state, epsilon)
        next_state, reward, done = env.step(action_idx)
        total_reward += reward
        state = next_state
        steps += 1
        if done:
            break

    return total_reward, steps

def main():
    env = SnakeGame()
    model = DQN(input_dim=env.grid_size * env.grid_size, output_dim=4)
    epsilon = EPS_START

    for episode in range(NUM_EPISODES):
        env.reset()  # Reset environment and snake position!
        reward, steps = run_episode(env, model, epsilon)
        print(f"Episode {episode+1}: Reward={reward}, Steps={steps}, Epsilon={epsilon:.3f}")
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

if __name__ == "__main__":
    main()
