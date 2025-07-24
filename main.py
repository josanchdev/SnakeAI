import pygame
import os
import torch
import numpy as np
from snake_game.game import SnakeGame
from agent.dqn import DQN

def get_latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
    return os.path.join(checkpoint_dir, files[-1])

def main():
    pygame.init()
    grid_size = 12
    cell_size = 32
    screen = pygame.display.set_mode((grid_size*cell_size, grid_size*cell_size))
    clock = pygame.time.Clock()
    pygame.display.set_caption("SlytherNN: Snake RL - Menu")

    # Adaptive font size
    font_size = max(20, int(min(screen.get_width(), screen.get_height()) // 20))
    font = pygame.font.SysFont("arial", font_size)
    menu_text = font.render("Press [Space] for AI, [Arrow Keys] for Human", True, (255,255,255))
    menu_rect = menu_text.get_rect(center=screen.get_rect().center)
    mode = None
    while mode is None:
        screen.fill((0,0,0))
        screen.blit(menu_text, menu_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if event.key == pygame.K_SPACE:
                    mode = "ai"
                elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    mode = "human"

    # Set up game and agent
    game = SnakeGame(grid_size, cell_size, mode=mode)
    ai_model = None
    if mode == "ai":
        checkpoint = get_latest_checkpoint("checkpoints")
        if checkpoint:
            ai_model = DQN(grid_size*grid_size, 4)
            ai_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            ai_model.eval()
        else:
            print("No checkpoint found. AI will not play.")

    MOVE_EVENT = pygame.USEREVENT
    pygame.time.set_timer(MOVE_EVENT, 90)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if mode == "human":
                    if event.key == pygame.K_UP:
                        game.snake.set_direction((0, -1))
                    elif event.key == pygame.K_DOWN:
                        game.snake.set_direction((0, 1))
                    elif event.key == pygame.K_LEFT:
                        game.snake.set_direction((-1, 0))
                    elif event.key == pygame.K_RIGHT:
                        game.snake.set_direction((1, 0))
                if event.key == pygame.K_r and not game.running:
                    game.reset()
        if event.type == MOVE_EVENT and game.running:
            if mode == "ai" and ai_model:
                state = game.get_state().flatten().astype(np.float32)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = ai_model(state_tensor)
                    action_idx = torch.argmax(q_values).item()
                game.ai_step(action_idx)
            else:
                game.update()

        game.draw(screen)
        if not game.running:
            game.draw_game_over(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
