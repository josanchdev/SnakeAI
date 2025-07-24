import os
import random
import numpy as np
import time
import torch
import torch.nn.functional as F
import csv

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from agent.dqn import DQN
from agent.memory import ReplayMemory
from snake_game.game import SnakeGame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==== Hyperparameters ====
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_EVERY = 5000 #
NUM_EPISODES = 10000   # Start small for debugging
MAX_STEPS_PER_EP = 100    # To prevent runaway loops
MEMORY_SIZE = 50000
BATCH_SIZE = 256
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

def optimize_model(model, memory, optimizer, batch_size, device, gamma=0.99):
    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.stack(states).to(device).view(batch_size, -1)
    actions     = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards     = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.stack(next_states).to(device).view(batch_size, -1)
    dones       = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    # Q(s, a)
    q_values = model(states).gather(1, actions)

    # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q_values = model(next_states).max(1, keepdim=True)[0]
    target_q = rewards + gamma * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def select_action(model, state, epsilon, device):
    # Choose random or best action
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    state = state.flatten().unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

def run_episode(env, model, epsilon, memory, optimizer, device):
    state = env.get_state(device)
    total_reward = 0
    env.running = True
    steps = 0
    cumulative_loss = 0
    update_count = 0
    stalled = False

    while env.running and steps < MAX_STEPS_PER_EP:
        action_idx = select_action(model, state, epsilon, device)
        next_state, reward, done = env.step(action_idx, device)
        memory.add((state, action_idx, reward, next_state, done))
        if len(memory) >= BATCH_SIZE:
            loss = optimize_model(model, memory, optimizer, BATCH_SIZE, device)
            cumulative_loss += loss
            update_count += 1
        total_reward += reward
        state = next_state
        steps += 1
        if done:
            break

    # If episode ended due to stalling (max steps), apply penalty
    if steps >= MAX_STEPS_PER_EP and env.running:
        stalled = True
        reward = -10
        done = True
        memory.add((state, action_idx, reward, state, done))
        total_reward += reward

    avg_loss = cumulative_loss / update_count if update_count > 0 else None
    return total_reward, steps, avg_loss

def main():
    start_time = time.time()  # ⏱️ Start timer

    env = SnakeGame()
    model = DQN(input_dim=env.grid_size * env.grid_size, output_dim=4).to(device)
    memory = ReplayMemory(MEMORY_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epsilon = EPS_START

    log_path = os.path.join(LOG_DIR, 'training_log.csv')
    checkpoint_path = lambda ep: os.path.join(CHECKPOINT_DIR, f'dqn_snake_checkpoint_ep{ep}.pth')

    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Steps', 'Epsilon', 'AvgLoss'])

        for episode in range(NUM_EPISODES):
            env.reset()
            reward, steps, avg_loss = run_episode(env, model, epsilon, memory, optimizer, device)
            writer.writerow([episode+1, reward, steps, epsilon, avg_loss if avg_loss is not None else 'N/A'])

            if (episode + 1) % SAVE_EVERY == 0:
                torch.save(model.state_dict(), checkpoint_path(episode + 1))

            epsilon = max(EPS_END, epsilon * EPS_DECAY)

    elapsed_time = time.time() - start_time  # ⏱️ End timer
    print(f"Training completed in {elapsed_time:.2f} seconds for {NUM_EPISODES} episodes.")

if __name__ == "__main__":
    print("Training started.")
    main()
    print("Training ended.")
