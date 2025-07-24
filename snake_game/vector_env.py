import torch
from snake_game.game import SnakeGame

class VectorEnv:
    def __init__(self, num_envs=64, grid_size=12, cell_size=32, device=None):
        self.num_envs = num_envs
        self.envs = [SnakeGame(grid_size=grid_size, cell_size=cell_size) for _ in range(num_envs)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        states = []
        for env in self.envs:
            env.reset()
            states.append(env.get_state(self.device))
        return torch.stack(states)

    def step(self, actions):
        # actions: tensor/list of action indices for each env
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            state, reward, done = env.step(action, self.device)
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
        return torch.stack(next_states), torch.tensor(rewards, device=self.device), torch.tensor(dones, device=self.device)

    def get_states(self):
        return torch.stack([env.get_state(self.device) for env in self.envs])

    def all_running(self):
        return [env.running for env in self.envs]
