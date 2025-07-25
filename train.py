import os
import random
import numpy as np
import time
import torch
import torch.nn.functional as F
import csv

from agent.dqn import DQN, ACTIONS
from agent.prioritized_memory import PrioritizedReplayMemory
from snake_game.vector_env import VectorEnv

# Seed and device setup
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_ENVS = 64
NUM_EPISODES = 4000
MAX_STEPS_PER_EP = 100
MEMORY_SIZE = 100000
BATCH_SIZE = 128
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

LEARNING_RATE = 5e-4
TARGET_UPDATE_FREQ = 1000  # steps
GRAD_ACCUM_STEPS = 2  # Number of optimize_model() calls before optimizer.step()

# Initialize environments in batch
envs = VectorEnv(num_envs=NUM_ENVS, device=device)
state_dim = envs.get_states().shape[1]

# Networks
policy_net = DQN(input_dim=state_dim, output_dim=4).to(device)
target_net = DQN(input_dim=state_dim, output_dim=4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = PrioritizedReplayMemory(capacity=MEMORY_SIZE)

# Mixed precision scaler (only if CUDA)
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

epsilon = EPS_START
step_count = 0  # total environment steps

def select_actions_batch(model, states, epsilon):
    """
    Batch epsilon-greedy action selection.

    Args:
        model: DQN
        states: (batch_size, state_dim)
        epsilon: float

    Returns:
        actions: torch.LongTensor (batch_size,)
    """
    batch_size = states.size(0)
    random_actions = torch.randint(0, len(ACTIONS), (batch_size,), device=states.device)
    with torch.no_grad():
        q_values = model(states)
        best_actions = torch.argmax(q_values, dim=1)
    probs = torch.rand(batch_size, device=states.device)
    chosen_actions = torch.where(probs < epsilon, random_actions, best_actions)
    return chosen_actions

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    batch, idxs, is_weights = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states      = torch.stack(states).to(device)
    actions     = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards     = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.stack(next_states).to(device)
    dones       = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    is_weights  = torch.tensor(is_weights, dtype=torch.float32, device=device).unsqueeze(1)

    if not hasattr(optimize_model, "accum_step"):
        optimize_model.accum_step = 0
    if not hasattr(optimize_model, "accum_loss"):
        optimize_model.accum_loss = 0.0

    if use_amp:
        with torch.cuda.amp.autocast():
            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q = target_net(next_states).max(1, keepdim=True)[0]
                expected_q = rewards + (0.99 * next_q * (1 - dones))
            td_errors = current_q - expected_q
            loss = (is_weights * td_errors.pow(2)).mean() / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
    else:
        current_q = policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_net(next_states).max(1, keepdim=True)[0]
            expected_q = rewards + (0.99 * next_q * (1 - dones))
        td_errors = current_q - expected_q
        loss = (is_weights * td_errors.pow(2)).mean() / GRAD_ACCUM_STEPS
        loss.backward()

    optimize_model.accum_step += 1
    optimize_model.accum_loss += loss.item()

    if optimize_model.accum_step % GRAD_ACCUM_STEPS == 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        optimize_model.accum_step = 0
        ret_loss = optimize_model.accum_loss
        optimize_model.accum_loss = 0.0
    else:
        ret_loss = None

    # Update priorities in PER
    td_errors_np = td_errors.detach().abs().cpu().numpy().flatten()
    memory.update_priorities(idxs, td_errors_np)
    return ret_loss

def main():
    global epsilon, step_count
    start_time = time.time()

    states = envs.reset()
    episode_rewards = torch.zeros(NUM_ENVS, device=device)
    episode_steps = torch.zeros(NUM_ENVS, device=device)
    episode_counts = torch.zeros(NUM_ENVS, device=device)
    rewards_log = []


    with open(os.path.join(LOG_DIR, 'training_log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Steps', 'Epsilon', 'AvgLoss'])

        while episode_counts.min() < NUM_EPISODES:
            actions = select_actions_batch(policy_net, states, epsilon)
            next_states, rewards, dones = envs.step(actions)

            # Store transitions
            for i in range(NUM_ENVS):
                memory.add((states[i].to(device), actions[i].item(), rewards[i].item(), next_states[i].to(device), dones[i].item()))

            # Accumulate rewards and steps
            episode_rewards += rewards
            episode_steps += 1

            # When done, log episode info and reset counters for that env
            for i in range(NUM_ENVS):
                if dones[i]:
                    print(f"Episode {int(episode_counts[i]+1)} finished in env {i} with reward {episode_rewards[i].item()}")
                    writer.writerow([int(episode_counts[i]+1), 
                                     episode_rewards[i].item(), 
                                     int(episode_steps[i].item()), 
                                     epsilon, 
                                     'NA'])  # you can add loss tracking if desired
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    episode_counts[i] += 1

            loss = optimize_model()

            # Update target network periodically
            if step_count % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Decay epsilon
            epsilon = max(EPS_END, epsilon * EPS_DECAY)

            states = next_states
            step_count += NUM_ENVS  # one step per env per loop

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds for {NUM_EPISODES} episodes.")

if __name__ == "__main__":
    print("Training started.")
    main()
    print("Training ended.")