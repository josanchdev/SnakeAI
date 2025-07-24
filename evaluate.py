import os
import torch
import numpy as np
import pygame
from snake_game.game import SnakeGame
from agent.dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
    return os.path.join(checkpoint_dir, files[-1])

def ai_play(num_episodes=10, grid_size=12, cell_size=32):
    checkpoint = get_latest_checkpoint("checkpoints")
    if not checkpoint:
        print("No checkpoint found.")
        return
    model = DQN(grid_size*grid_size, 4)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    epsilon = 0.0  # Fully greedy

    pygame.init()
    screen = pygame.display.set_mode((grid_size*cell_size, grid_size*cell_size))
    clock = pygame.time.Clock()
    pygame.display.set_caption("SlytherNN: AI Evaluation")

    for episode in range(num_episodes):
        game = SnakeGame(grid_size, cell_size, mode="ai")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); return
            if game.running:
                state = game.get_state().flatten().astype(np.float32)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action_idx = torch.argmax(q_values).item()
                game.ai_step(action_idx)
            game.draw(screen)
            if not game.running:
                game.draw_game_over(screen)
            pygame.display.flip()
            clock.tick(60)
        print(f"Episode {episode+1}: Score={game.score}")
    pygame.quit()

if __name__ == "__main__":
    ai_play(num_episodes=10)
