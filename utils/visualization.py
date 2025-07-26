# utils/visualization.py
"""
Visualization utilities for training progress and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from config import PLOTS_DIR
import os


def plot_training_progress(metrics: List[Dict], save_name: str = "training_progress.png", 
                          window: int = 100):
    """Create comprehensive training progress plots."""
    df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SlytherNN Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Reward over time (smoothed)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('GlobalEpisode')
    smoothed_reward = df_sorted['Reward'].rolling(window, min_periods=1).mean()
    ax1.plot(df_sorted['GlobalEpisode'], smoothed_reward, color='#2E86C1', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Average Reward (smoothed, window={window})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Steps per episode (smoothed)
    ax2 = axes[0, 1]
    smoothed_steps = df_sorted['Steps'].rolling(window, min_periods=1).mean()
    ax2.plot(df_sorted['GlobalEpisode'], smoothed_steps, color='#28B463', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title(f'Average Steps per Episode (smoothed, window={window})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss over time (smoothed)
    ax3 = axes[1, 0]
    smoothed_loss = df_sorted['AvgLoss'].rolling(window, min_periods=1).mean()
    ax3.plot(df_sorted['GlobalEpisode'], smoothed_loss, color='#E74C3C', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'Average Loss (smoothed, window={window})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Epsilon decay
    ax4 = axes[1, 1]
    ax4.plot(df_sorted['GlobalEpisode'], df_sorted['Epsilon'], color='#8E44AD', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.set_title('Exploration Rate (Epsilon)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_distribution(metrics: List[Dict], save_name: str = "performance_dist.png",
                                last_pct: float = 0.1):
    """Plot performance distribution for the last portion of training."""
    df = pd.DataFrame(metrics)
    
    # Get last portion of episodes
    max_episode = df['GlobalEpisode'].max()
    cutoff = int(max_episode * (1 - last_pct))
    recent_df = df[df['GlobalEpisode'] > cutoff]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Performance Distribution (Last {last_pct*100:.0f}% of Training)', 
                 fontsize=14, fontweight='bold')
    
    # Reward distribution
    ax1 = axes[0]
    sns.histplot(data=recent_df, x='Reward', bins=30, kde=True, ax=ax1, color='#3498DB')
    ax1.axvline(recent_df['Reward'].mean(), color='red', linestyle='--', 
                label=f'Mean: {recent_df["Reward"].mean():.2f}')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    
    # Steps distribution
    ax2 = axes[1]
    sns.histplot(data=recent_df, x='Steps', bins=30, kde=True, ax=ax2, color='#27AE60')
    ax2.axvline(recent_df['Steps'].mean(), color='red', linestyle='--',
                label=f'Mean: {recent_df["Steps"].mean():.1f}')
    ax2.set_title('Steps Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()


def generate_training_summary(metrics: List[Dict]) -> Dict:
    """Generate summary statistics from training metrics."""
    df = pd.DataFrame(metrics)
    
    # Overall statistics
    summary = {
        'total_episodes': len(df),
        'training_duration': df['GlobalEpisode'].max(),
        'final_epsilon': df['Epsilon'].iloc[-1] if len(df) > 0 else 0,
        
        # Performance metrics
        'mean_reward': df['Reward'].mean(),
        'max_reward': df['Reward'].max(),
        'min_reward': df['Reward'].min(),
        'std_reward': df['Reward'].std(),
        
        'mean_steps': df['Steps'].mean(),
        'max_steps': df['Steps'].max(),
        'min_steps': df['Steps'].min(),
        
        'mean_loss': df['AvgLoss'].mean(),
        'final_loss': df['AvgLoss'].iloc[-1] if len(df) > 0 else 0,
    }
    
    # Performance in last 10% of training
    if len(df) > 100:  # Only if we have enough data
        last_10pct = df.tail(len(df) // 10)
        summary.update({
            'recent_mean_reward': last_10pct['Reward'].mean(),
            'recent_mean_steps': last_10pct['Steps'].mean(),
            'recent_mean_loss': last_10pct['AvgLoss'].mean(),
        })
    
    return summary


def create_gif_from_gameplay(frames: List[np.ndarray], save_path: str, fps: int = 10):
    """Create a GIF from gameplay frames."""
    try:
        from PIL import Image
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
        
        # Save as GIF
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # Duration in milliseconds
            loop=0
        )
        
        print(f"GIF saved to: {save_path}")
        
    except ImportError:
        print("PIL not available. Install with: pip install Pillow")
    except Exception as e:
        print(f"Error creating GIF: {e}")


def plot_reward_comparison(log_files: List[str], labels: List[str], 
                          save_name: str = "reward_comparison.png", window: int = 100):
    """Compare training progress across multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD']
    
    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        try:
            df = pd.read_csv(os.path.join(LOG_DIR, log_file))
            df_sorted = df.sort_values('GlobalEpisode')
            smoothed_reward = df_sorted['Reward'].rolling(window, min_periods=1).mean()
            
            ax.plot(df_sorted['GlobalEpisode'], smoothed_reward, 
                   color=colors[i % len(colors)], linewidth=2, label=label)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Training Progress Comparison (smoothed, window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()