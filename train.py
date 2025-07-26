#!/usr/bin/env python3
"""
SlytherNN Training Script

High-performance DQN training for Snake with vectorized environments.
Optimized for RTX 3090 + Ryzen 9 5900X hardware configuration.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from agent.dqn import DQN
from agent.prioritized_memory import PrioritizedReplayMemory
from snake_game.vector_env import VectorEnv
from config import Config, DEVICE, INPUT_DIM, OUTPUT_DIM, CHECKPOINTS_DIR, LOGS_DIR
from utils.logging import TrainingLogger, CheckpointManager
from utils.visualization import plot_training_progress


class SlytherNNTrainer:
    """High-performance DQN trainer for Snake."""
    
    def __init__(self):
        self.config = Config()
        self._setup_reproducibility()
        self._setup_training_components()
        self._setup_logging()
        print(f"🚀 SlytherNN Trainer initialized on {DEVICE}")
    
    def _setup_reproducibility(self):
        """Set random seeds for reproducible training."""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED) 
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
            torch.backends.cudnn.deterministic = True
    
    def _setup_training_components(self):
        """Initialize networks, optimizer, and training utilities."""
        # Vectorized environments
        self.envs = VectorEnv(self.config.NUM_ENVS, device=DEVICE)
        
        # Neural networks with larger architecture
        self.policy_net = DQN(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dims=self.config.HIDDEN_DIMS,
            dropout=self.config.DROPOUT
        ).to(DEVICE)
        
        self.target_net = DQN(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM, 
            hidden_dims=self.config.HIDDEN_DIMS,
            dropout=self.config.DROPOUT
        ).to(DEVICE)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50000, gamma=0.9
        )
        
        # Experience replay with PER
        self.memory = PrioritizedReplayMemory(
            capacity=self.config.MEMORY_SIZE,
            alpha=self.config.PER_ALPHA,
            beta=self.config.PER_BETA_START
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Training state
        self.epsilon = self.config.EPS_START
        self.steps_done = 0
        self.episodes_done = 0
        self.best_mean_reward = float('-inf')
    
    def _setup_logging(self):
        """Initialize logging and checkpoint management."""
        self.logger = TrainingLogger()
        self.checkpoint_manager = CheckpointManager(max_checkpoints=5)
        
        # Load checkpoint if available
        checkpoint_path, episodes = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, episodes)
    
    def _load_checkpoint(self, path: str, episodes: int):
        """Load training state from checkpoint."""
        print(f"📁 Loading checkpoint: {path}")
        
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.episodes_done = episodes
        self.epsilon = checkpoint.get('epsilon', self.config.EPS_START)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.best_mean_reward = checkpoint.get('best_mean_reward', float('-inf'))
        
        print(f"✅ Resumed from episode {episodes}")
    
    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Vectorized epsilon-greedy action selection."""
        batch_size = states.size(0)
        
        with torch.no_grad():
            # Get Q-values from policy network
            q_values = self.policy_net(states)
            greedy_actions = q_values.argmax(dim=1)
        
        # Epsilon-greedy exploration
        random_mask = torch.rand(batch_size, device=DEVICE) < self.epsilon
        random_actions = torch.randint(0, OUTPUT_DIM, (batch_size,), device=DEVICE)
        
        actions = torch.where(random_mask, random_actions, greedy_actions)
        return actions
    
    def optimize_model(self) -> float:
        """Perform one optimization step."""
        if len(self.memory) < self.config.BATCH_SIZE:
            return 0.0
        
        # Sample batch from PER
        batch, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        # Compute loss with mixed precision
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                loss, td_errors = self._compute_dqn_loss(
                    states, actions, rewards, next_states, dones, weights
                )
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, td_errors = self._compute_dqn_loss(
                states, actions, rewards, next_states, dones, weights
            )
            loss.backward()
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # Update PER priorities
        priorities = td_errors.abs().detach().cpu().numpy() + self.config.PER_EPS
        self.memory.update_priorities(indices, priorities)
        
        return loss.item()
    
    def _compute_dqn_loss(self, states, actions, rewards, next_states, dones, weights):
        """Compute Double DQN loss with importance sampling."""
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use policy net to select actions, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (
                self.config.GAMMA * next_q_values * (~dones).unsqueeze(1)
            )
        
        # Compute TD errors and weighted loss
        td_errors = current_q_values - target_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        return loss, td_errors.squeeze()
    
    def update_target_network(self):
        """Soft update of target network."""
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                             self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.TAU * policy_param.data + 
                (1.0 - self.config.TAU) * target_param.data
            )
    
    def decay_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(
            self.config.EPS_END,
            self.epsilon * self.config.EPS_DECAY
        )
    
    def save_checkpoint(self):
        """Save current training state."""
        checkpoint_data = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'best_mean_reward': self.best_mean_reward,
            'config': self.config.__dict__,
        }
        
        return self.checkpoint_manager.save_checkpoint(
            checkpoint_data, self.episodes_done
        )
    
    def train(self):
        """Main training loop."""
        print("🎯 Starting SlytherNN training...")
        
        states = self.envs.reset()
        episode_rewards = torch.zeros(self.config.NUM_ENVS, device=DEVICE)
        episode_lengths = torch.zeros(self.config.NUM_ENVS, device=DEVICE)
        recent_losses = []
        
        start_time = time.time()
        last_print_time = start_time
        
        while self.episodes_done < self.config.NUM_EPISODES:
            # Select and execute actions
            actions = self.select_actions(states)
            next_states, rewards, dones = self.envs.step(actions)
            
            # Store transitions
            for i in range(self.config.NUM_ENVS):
                self.memory.add((
                    states[i].cpu(),
                    actions[i].item(),
                    rewards[i].item(),
                    next_states[i].cpu(),
                    dones[i].item()
                ))
            
            # Update tracking
            episode_rewards += rewards
            episode_lengths += 1
            states = next_states
            self.steps_done += self.config.NUM_ENVS
            
            # Optimize model
            if self.steps_done % self.config.UPDATE_EVERY == 0:
                loss = self.optimize_model()
                recent_losses.append(loss)
                
                # Keep only recent losses
                if len(recent_losses) > 1000:
                    recent_losses = recent_losses[-1000:]
            
            # Update target network
            if self.steps_done % self.config.TARGET_UPDATE_FREQ == 0:
                self.update_target_network()
            
            # Handle completed episodes
            for i in range(self.config.NUM_ENVS):
                if dones[i]:
                    self.episodes_done += 1
                    
                    # Log episode
                    self.logger.log_episode(
                        global_episode=self.episodes_done,
                        env_id=i,
                        episode=self.episodes_done,
                        reward=episode_rewards[i].item(),
                        steps=int(episode_lengths[i].item()),
                        epsilon=self.epsilon,
                        avg_loss=np.mean(recent_losses) if recent_losses else 0.0
                    )
                    
                    # Reset episode tracking
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
            
            # Decay exploration
            self.decay_epsilon()
            
            # Periodic logging and saving
            current_time = time.time()
            if current_time - last_print_time >= 60:  # Every minute
                self._print_progress(current_time - start_time, recent_losses)
                last_print_time = current_time
            
            if self.episodes_done % self.config.SAVE_EVERY == 0 and self.episodes_done > 0:
                self.save_checkpoint()
        
        # Training completed
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time/3600:.2f} hours!")
        
        # Generate final visualizations
        self._finalize_training()
    
    def _print_progress(self, elapsed_time: float, recent_losses: list):
        """Print training progress."""
        if not self.logger.metrics_history:
            return
        
        recent_metrics = self.logger.metrics_history[-100:]  # Last 100 episodes
        avg_reward = np.mean([m['reward'] for m in recent_metrics])
        avg_steps = np.mean([m['steps'] for m in recent_metrics])
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        print(f"Episode {self.episodes_done:5d} | "
              f"Reward: {avg_reward:6.2f} | "
              f"Steps: {avg_steps:5.1f} | "  
              f"Loss: {avg_loss:.4f} | "
              f"ε: {self.epsilon:.3f} | "
              f"Time: {elapsed_time/3600:.2f}h")
    
    def _finalize_training(self):
        """Generate final results and cleanup."""
        print("📊 Generating training visualizations...")
        
        if self.logger.metrics_history:
            # Create plots
            plot_training_progress(self.logger.metrics_history)
            
            # Save final checkpoint
            self.save_checkpoint()
            
            # Calculate final statistics
            final_metrics = self.logger.metrics_history[-1000:]  # Last 1000 episodes
            final_reward = np.mean([m['reward'] for m in final_metrics])
            
            print(f"✅ Final mean reward: {final_reward:.2f}")
            print(f"📁 Results saved to {PLOTS_DIR}")


def main():
    """Entry point for training."""
    try:
        trainer = SlytherNNTrainer()
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()