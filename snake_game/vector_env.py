
import torch
from snake_game.game import SnakeGame
from concurrent.futures import ThreadPoolExecutor

class VectorEnv:
    def __init__(self, num_envs=64, grid_size=12, cell_size=32, device=None):
        self.num_envs = num_envs
        self.envs = [SnakeGame(grid_size=grid_size, cell_size=cell_size) for _ in range(num_envs)]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        """
        Reset all environments and return batched initial states.
        Returns:
            states: torch.Tensor (num_envs, state_dim)
        """
        for env in self.envs:
            env.reset()
        return self.get_states()

    def step(self, actions):
        """
        Step all environments with the given batch of actions in parallel (threaded).
        Automatically resets environments that are done before stepping.

        Args:
            actions: torch.Tensor or list/array of ints (num_envs,)

        Returns:
            next_states: torch.Tensor (num_envs, state_dim)
            rewards: torch.Tensor (num_envs,)
            dones: torch.Tensor (num_envs,)
        """
        def step_env(env, action):
            if not env.running:
                env.reset()
            return env.step(int(action), self.device)

        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            results = list(executor.map(step_env, self.envs, actions))

        next_states, rewards, dones = zip(*results)
        return (
            torch.stack(next_states),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.bool)
        )

    def get_states(self):
        """
        Get current states from all environments as a batched tensor.
        Returns:
            states: torch.Tensor (num_envs, state_dim)
        """
        return torch.stack([env.get_state(self.device) for env in self.envs])

    def all_running(self):
        return [env.running for env in self.envs]
