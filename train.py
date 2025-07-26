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

NUM_ENVS = 256
NUM_EPISODES = 1000
MEMORY_SIZE = 100000
BATCH_SIZE = 256
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
MAX_STEPS_SINCE_REWARD = 100  
LEARNING_RATE = 5e-4
TARGET_UPDATE_FREQ = 1000  # steps
GRAD_ACCUM_STEPS = 2  # Number of optimize_model() calls before optimizer.step()
SAVE_EVERY = 1000  # Save every N completed episodes
STALL_PENALTY = -15

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
scaler = torch.amp.GradScaler('cuda') if use_amp else None

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
        with torch.amp.autocast('cuda'):
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


def get_latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'dqn_snake_checkpoint_ep' in f]
    if not files:
        return None, 0
    files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
    latest = files[-1]
    ep = int(latest.split('_ep')[1].split('.pth')[0])
    return os.path.join(checkpoint_dir, latest), ep

def main():
    global epsilon, step_count
    start_time = time.time()

    # Resume from latest checkpoint if available
    checkpoint, completed_episodes = get_latest_checkpoint(CHECKPOINT_DIR)
    episode_rewards = torch.zeros(NUM_ENVS, device=device)
    episode_steps = torch.zeros(NUM_ENVS, device=device)
    episode_counts = torch.zeros(NUM_ENVS, device=device)
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        per_env_missing = False
        if isinstance(state, dict) and 'model' in state and 'optimizer' in state:
            policy_net.load_state_dict(state['model'])
            target_net.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            # Restore per-env stats if present
            if 'episode_counts' in state:
                episode_counts = torch.tensor(state['episode_counts'], device=device)
            else:
                per_env_missing = True
            if 'episode_rewards' in state:
                episode_rewards = torch.tensor(state['episode_rewards'], device=device)
            else:
                per_env_missing = True
            if 'episode_steps' in state:
                episode_steps = torch.tensor(state['episode_steps'], device=device)
            else:
                per_env_missing = True
            if per_env_missing:
                print("[WARN] Checkpoint missing per-env episode_counts/rewards/steps, initializing all to zero and starting per-env logs from 0.")
        else:
            policy_net.load_state_dict(state)
            target_net.load_state_dict(state)
        print(f"Resumed at {completed_episodes} completed episodes.")
    else:
        completed_episodes = 0

    states = envs.reset()
    rewards_log = []
    # New: Track steps since last reward increase for each env
    steps_since_last_reward = torch.zeros(NUM_ENVS, device=device)
    last_rewards = torch.zeros(NUM_ENVS, device=device)

    def checkpoint_path(ep):
        return os.path.join(CHECKPOINT_DIR, f'dqn_snake_checkpoint_ep{ep}.pth')

    def cleanup_old_checkpoints(current_ep, keep=3):
        """Remove checkpoints older than the most recent `keep` ones."""
        all_ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('dqn_snake_checkpoint_ep') and f.endswith('.pth')]
        # Extract episode numbers
        eps = []
        for fname in all_ckpts:
            try:
                num = int(fname.split('_ep')[1].split('.pth')[0])
                eps.append((num, fname))
            except Exception:
                continue
        eps.sort()
        # Remove all but the last `keep` checkpoints
        for num, fname in eps[:-keep]:
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fname))
            except Exception:
                pass


    import datetime
    with open(os.path.join(LOG_DIR, 'training_log.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Prepare header: add reward breakdown columns if available in future
        header = [
            'GlobalEpisode', 'EnvID', 'Episode', 'Reward', 'Steps', 'Epsilon', 'AvgLoss', 'Timestamp'
            # 'Reward_Fruit', 'Reward_Step', ... (future)
        ]
        writer.writerow(header)

        # Track per-episode losses for each env
        episode_losses = torch.zeros(NUM_ENVS, device=device)
        global_episode_counter = 0
        last_milestone = 0


        while episode_counts.min() < NUM_EPISODES:
            actions = select_actions_batch(policy_net, states, epsilon)
            next_states, rewards, dones = envs.step(actions)

            # New: Update steps_since_last_reward
            for i in range(NUM_ENVS):
                # If reward increased, reset counter
                if rewards[i].item() > last_rewards[i].item():
                    steps_since_last_reward[i] = 0
                else:
                    steps_since_last_reward[i] += 1
                last_rewards[i] = rewards[i]

            # End episode if steps_since_last_reward exceeds threshold
            for i in range(NUM_ENVS):
                if steps_since_last_reward[i] >= MAX_STEPS_SINCE_REWARD:
                    dones[i] = True
                    rewards[i] = STALL_PENALTY

            # Store transitions
            for i in range(NUM_ENVS):
                memory.add((states[i].to(device), actions[i].item(), rewards[i].item(), next_states[i].to(device), dones[i].item()))

            # Accumulate rewards, steps, and losses
            episode_rewards += rewards
            episode_steps += 1

            # Get loss for this step (shared for all envs, so just use for all)
            loss = optimize_model()
            if loss is not None:
                # Distribute loss equally to all running envs for this step
                for i in range(NUM_ENVS):
                    if not dones[i]:
                        episode_losses[i] += loss / NUM_ENVS

            # When done, log episode info and reset counters for that env
            for i in range(NUM_ENVS):
                if dones[i]:
                    global_episode_counter += 1
                    completed_episodes += 1
                    avg_loss = episode_losses[i].item() / episode_steps[i].item() if episode_steps[i].item() > 0 else 0.0
                    timestamp = datetime.datetime.now().isoformat()
                    writer.writerow([
                        global_episode_counter,  # GlobalEpisode
                        i,                      # EnvID
                        int(episode_counts[i]+1), # Episode (per-env)
                        episode_rewards[i].item(),
                        int(episode_steps[i].item()),
                        epsilon,
                        avg_loss,
                        timestamp
                        # reward_fruit, reward_step, ... (future)
                    ])
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    episode_counts[i] += 1
                    episode_losses[i] = 0
                    steps_since_last_reward[i] = 0
                    last_rewards[i] = 0
                    # Save checkpoint every SAVE_EVERY completed episodes
                    if completed_episodes % SAVE_EVERY == 0:
                        ckpt_path = checkpoint_path(completed_episodes)
                        torch.save({
                            'model': policy_net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'episode_counts': episode_counts.cpu().numpy(),
                            'episode_rewards': episode_rewards.cpu().numpy(),
                            'episode_steps': episode_steps.cpu().numpy(),
                        }, ckpt_path)
                        cleanup_old_checkpoints(completed_episodes, keep=3)

            # Print milestone summary only once per 100 global episodes (not per env, not at 0)
            min_episode = int(episode_counts.min().item())
            if min_episode >= last_milestone + 100 and min_episode > 0:
                last_milestone = (min_episode // 100) * 100
                print(f"Milestone: {last_milestone} episodes completed.")

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