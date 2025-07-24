# 🐍 SlytherNN: Deep RL Snake on GPU

## Overview
A Deep Q-Network (DQN) agent trained to play Snake, optimized for GPU acceleration with PyTorch. This project showcases performance engineering, reproducibility, and modern RL techniques, serving as a strong portfolio piece for AI internships and research.

---

## Table of Contents
- Features
- Setup
- How to Run & Test Yourself
- Performance & Optimization Results
- Technical Details
- Changelog & Speedup Log
- System Requirements
- Struggles & Solutions

---

## Features
- DQN-based Snake Agent (CPU/GPU support)
- Modular, extensible codebase
- GPU-optimized training loop (torch.Tensor, batching)
- Automatic logging (scores, training curves)
- Playable UI (human/AI modes)
- Performance benchmarking (timing, nvidia-smi integration)

---

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/josanchdev/SlytherNN.git
cd SlytherNN
uv pip install -r requirements.txt
```

---

## How to Run & Test Yourself

```bash
# Train the agent (uses GPU if available)
uv run train.py

# Run tests
pytest tests/
```

---

## Performance & Optimization Results

| Change                | Speedup | Notes                                  |
|-----------------------|---------|----------------------------------------|
| GPU tensor batching   | +3 sec  | Switched from NumPy to torch.Tensor    |
| Larger DQN model      |         | Utilizes RTX 3090 more efficiently     |
| Batched environments  |         | (Planned)                              |
| AMP/torch.compile     |         | (Planned)                              |

---

## Technical Details

- **RL Algorithm:** Deep Q-Network (DQN)
- **Model:** Linear(144→1024)→ReLU→1024→ReLU→512→ReLU→256→4
- **Batch Size:** 1024 (for full GPU utilization)
- **Replay Memory:** Experience replay for stability
- **Device Management:** All tensors/models on CUDA if available
- **Logging:** CSV logs, score plots

---

## Changelog & Speedup Log

- Refactored get_state() to return torch.Tensor on GPU
- Removed NumPy bottlenecks in training loop
- Updated all code/tests for device compatibility
- Increased model size for GPU utilization
- (Planned) Vectorized environment for parallel training
- (Planned) AMP/torch.compile for further speedup

---

## System Requirements

- Python 3.11+
- PyTorch (CUDA 12+ recommended)
- Pygame
- RTX 3090 or similar for best results

---

## Struggles & Solutions

| Issue                                      | Solution                                      |
|---------------------------------------------|-----------------------------------------------|
| PyTorch not using GPU                       | Explicit device placement for model/tensors   |
| NumPy/torch incompatibility (CUDA tensors)  | Switched to torch.stack for batching          |
| Tests failing after refactor                | Updated all calls to pass device argument     |
| Model too small for GPU                     | Deepened and widened DQN architecture         |
| Training loop underutilizing GPU            | (Planned) Batched environments, larger batch  |

---

## Credits

- [Your Name]
- [Mentors, Collaborators, etc.]

---

This structure will help you track your technical journey, highlight optimizations, and document struggles/solutions for future reference or professional review. You can copy, expand, and update this as your project evolves!
