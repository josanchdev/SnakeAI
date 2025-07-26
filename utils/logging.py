# utils/logging.py
"""
Logging utilities for training metrics and model checkpoints.
"""

import os
import csv
import json
import datetime
import torch
from typing import Dict, List, Any, Optional, Tuple
from config import CHECKPOINTS_DIR, LOGS_DIR  # Fixed import


class TrainingLogger:
    """Handles logging of training metrics to CSV and console."""
    
    def __init__(self, log_file: str = "training_log.csv"):
        self.log_path = os.path.join(LOGS_DIR, log_file)
        self.metrics_history = []
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'GlobalEpisode', 'EnvID', 'Episode', 'Reward', 'Steps', 
                'Epsilon', 'AvgLoss', 'Timestamp'
            ]
            writer.writerow(header)
    
    def log_episode(self, global_episode: int, env_id: int, episode: int, 
                   reward: float, steps: int, epsilon: float, avg_loss: float):
        """Log a single episode's metrics."""
        timestamp = datetime.datetime.now().isoformat()
        
        with open(self.log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                global_episode, env_id, episode, reward, steps, 
                epsilon, avg_loss, timestamp
            ])
        
        # Store in memory for analysis
        self.metrics_history.append({
            'global_episode': global_episode,
            'env_id': env_id,
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'avg_loss': avg_loss,
            'timestamp': timestamp
        })
    
    def log_milestone(self, episode: int, avg_reward: float, avg_steps: float, 
                     avg_loss: float, epsilon: float, elapsed_time: float):
        """Log milestone progress to console."""
        print(f"Episode {episode:4d} | "
              f"Avg Reward: {avg_reward:6.2f} | "
              f"Avg Steps: {avg_steps:5.1f} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Time: {elapsed_time:.1f}s")


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup."""
    
    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
    
    def save_checkpoint(self, model_state: Dict[str, Any], episode: int, 
                       additional_data: Optional[Dict] = None) -> str:
        """Save model checkpoint with metadata."""
        checkpoint_data = {
            'model': model_state,
            'episode': episode,
            'timestamp': datetime.datetime.now().isoformat(),
            **(additional_data or {})
        }
        
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'dqn_snake_checkpoint_ep{episode}.pth')
        torch.save(checkpoint_data, ckpt_path)
        
        self._cleanup_old_checkpoints()
        return ckpt_path
    
    def load_latest_checkpoint(self) -> Tuple[Optional[str], int]:
        """Load the most recent checkpoint."""
        if not os.path.exists(CHECKPOINTS_DIR):
            return None, 0
            
        files = [f for f in os.listdir(CHECKPOINTS_DIR) 
                if f.endswith('.pth') and 'dqn_snake_checkpoint_ep' in f]
        
        if not files:
            return None, 0
        
        # Sort by episode number
        files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
        latest = files[-1]
        episode = int(latest.split('_ep')[1].split('.pth')[0])
        
        return os.path.join(CHECKPOINTS_DIR, latest), episode
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        all_ckpts = [f for f in os.listdir(CHECKPOINTS_DIR) 
                    if f.startswith('dqn_snake_checkpoint_ep') and f.endswith('.pth')]
        
        if len(all_ckpts) <= self.max_checkpoints:
            return
        
        # Parse episode numbers and sort
        episodes = []
        for fname in all_ckpts:
            try:
                num = int(fname.split('_ep')[1].split('.pth')[0])
                episodes.append((num, fname))
            except (IndexError, ValueError):
                continue
        
        episodes.sort()
        
        # Remove oldest checkpoints
        for num, fname in episodes[:-self.max_checkpoints]:
            try:
                os.remove(os.path.join(CHECKPOINTS_DIR, fname))
            except OSError:
                pass  # File might have been deleted already


def save_training_config(config_dict: Dict[str, Any]):
    """Save training configuration for reproducibility."""
    config_path = os.path.join(LOGS_DIR, 'training_config.json')
    
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        if hasattr(value, '__dict__'):
            serializable_config[key] = vars(value)
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_training_metrics(log_file: str = "training_log.csv") -> List[Dict]:
    """Load training metrics from CSV file."""
    log_path = os.path.join(LOGS_DIR, log_file)
    
    if not os.path.exists(log_path):
        return []
    
    metrics = []
    with open(log_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric columns
            for col in ['GlobalEpisode', 'EnvID', 'Episode', 'Reward', 'Steps']:
                row[col] = int(row[col])
            for col in ['Epsilon', 'AvgLoss']:
                row[col] = float(row[col])
            metrics.append(row)
    
    return metrics