"""
SlytherNN Configuration

Centralized configuration for training, evaluation, and model parameters.
Optimized for RTX 3090 (24GB) + Ryzen 9 5900X (32GB RAM) setup.
"""

import os
import torch

# Hardware-optimized settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_MEMORY_GB = 24  # RTX 3090
SYSTEM_RAM_GB = 32   # Available RAM
CPU_CORES = 12       # Ryzen 9 5900X

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints") 
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
for directory in [LOGS_DIR, CHECKPOINTS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Game settings
GRID_SIZE = 12
CELL_SIZE = 32

# Training configuration - optimized for your hardware
class Config:
    # Environment settings (optimized for 24GB VRAM)
    NUM_ENVS = 128        # Increased from 64 for better GPU utilization
    MAX_STEPS_PER_EP = 200  # Increased for longer episodes
    
    # Training parameters  
    NUM_EPISODES = 4000
    BATCH_SIZE = 256      # Larger batch for RTX 3090
    MEMORY_SIZE = 200000  # Larger replay buffer (32GB RAM)
    
    # Learning parameters
    LEARNING_RATE = 3e-4  # Slightly reduced for stability
    GAMMA = 0.99
    TAU = 0.005          # Soft update rate for target network
    
    # Exploration schedule
    EPS_START = 1.0
    EPS_END = 0.01       # Lower final epsilon
    EPS_DECAY = 0.9995   # Slower decay for longer exploration
    
    # Training efficiency
    TARGET_UPDATE_FREQ = 500   # More frequent updates
    GRAD_ACCUM_STEPS = 1       # No accumulation needed with larger batch
    UPDATE_EVERY = 4           # Update frequency
    
    # Logging and checkpointing
    SAVE_EVERY = 500          # More frequent saves
    LOG_EVERY = 100
    PRINT_EVERY = 1000        # Console updates
    
    # Model architecture
    HIDDEN_DIMS = [256, 256]  # Larger network
    DROPOUT = 0.1
    
    # Reward structure
    REWARD_FRUIT = 10.0       # Higher fruit reward
    REWARD_STEP = -0.01
    REWARD_DEATH = -10.0
    REWARD_WIN = 200.0        # Higher win bonus
    
    # Reproducibility
    SEED = 42
    
    # PER parameters
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_STEPS = 100000
    PER_EPS = 1e-6

# Model dimensions
INPUT_DIM = GRID_SIZE * GRID_SIZE + 4 + 2  # grid + direction + relative_fruit
OUTPUT_DIM = 4  # number of actions

# Evaluation settings
class EvalConfig:
    NUM_EPISODES = 20
    MAX_STEPS = 1000
    EPSILON = 0.0        # Fully greedy
    RENDER_FPS = 8       # For GIF creation
    GIF_DURATION = 10    # Max seconds for GIF

# File patterns for cleanup
CHECKPOINT_PATTERN = "dqn_snake_checkpoint_ep*.pth"
LOG_PATTERN = "training_log*.csv"

def get_device_info():
    """Get detailed device information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{gpu_name} ({gpu_memory:.1f}GB)"
    return "CPU"

def print_config():
    """Print current configuration."""
    print("🐍 SlytherNN Configuration")
    print("=" * 40)
    print(f"Device: {get_device_info()}")
    print(f"Environments: {Config.NUM_ENVS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Memory Size: {Config.MEMORY_SIZE:,}")
    print(f"Episodes: {Config.NUM_EPISODES:,}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print("=" * 40)