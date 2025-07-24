import pygame
import sys
import numpy as np
import torch
from snake_game.utils import random_position

class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.body = [(grid_size//2, grid_size//2), (grid_size//2-1, grid_size//2), (grid_size//2-2, grid_size//2)]
        self.direction = (1, 0)
        self.grow = False

    def move(self):
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        if self.grow:
            self.body = [new_head] + self.body
            self.grow = False
        else:
            self.body = [new_head] + self.body[:-1]

    def set_direction(self, dir_tuple):
        dx, dy = dir_tuple
        if (dx, dy) == (-self.direction[0], -self.direction[1]):
            return
        self.direction = dir_tuple

    def grow_snake(self):
        self.grow = True

    def head(self):
        return self.body[0]

    def collided_with_self(self):
        return self.body[0] in self.body[1:]

    def collided_with_wall(self):
        x, y = self.body[0]
        return x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size

class Fruit:
    def __init__(self, grid_size, snake_body):
        self.grid_size = grid_size
        self.position = self.new_position(snake_body)

    def new_position(self, snake_body):
        while True:
            pos = random_position(self.grid_size)
            if pos not in snake_body:
                return pos

    def respawn(self, snake_body):
        self.position = self.new_position(snake_body)

class SnakeGame:
    def __init__(self, grid_size=12, cell_size=32, mode="human",
                 reward_fruit=1, reward_death=-10, reward_step=-0.01):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.snake = Snake(grid_size)
        self.fruit = Fruit(grid_size, self.snake.body)
        self.score = 0
        self.running = True
        self.mode = mode  # "human" or "ai"
        self.reward_fruit = reward_fruit
        self.reward_death = reward_death
        self.reward_step = reward_step

    def ai_step(self, action_idx, device):
        """Step using AI action index."""
        self.step(action_idx, device)

    def update(self):
        self.snake.move()
        if self.snake.head() == self.fruit.position:
            self.snake.grow_snake()
            self.fruit.respawn(self.snake.body)
            self.score += 1
        if self.snake.collided_with_self() or self.snake.collided_with_wall():
            self.running = False

    def draw(self, screen, board_offset_x=0, board_offset_y=0):
        # Modern dark gradient background
        for y in range(screen.get_height()):
            color = (
                26 + int(20 * y / screen.get_height()),
                26 + int(20 * y / screen.get_height()),
                32 + int(40 * y / screen.get_height())
            )
            pygame.draw.line(screen, color, (0, y), (screen.get_width(), y))

        # Draw grid borders
        grid_w = self.grid_size * self.cell_size
        grid_h = self.grid_size * self.cell_size
        border_rect = pygame.Rect(board_offset_x, board_offset_y, grid_w, grid_h)
        pygame.draw.rect(screen, (80, 80, 100), border_rect, width=4, border_radius=12)

        # Draw snake
        for segment in self.snake.body:
            pygame.draw.rect(
                screen, (0, 180, 45),
                (
                    board_offset_x + segment[0]*self.cell_size,
                    board_offset_y + segment[1]*self.cell_size,
                    self.cell_size, self.cell_size
                ),
                border_radius=7
            )
        # Draw fruit
        fx, fy = self.fruit.position
        pygame.draw.ellipse(
            screen, (220, 60, 60),
            (
                board_offset_x + fx*self.cell_size,
                board_offset_y + fy*self.cell_size,
                self.cell_size, self.cell_size
            ),
        )
        self.draw_scoreboard(screen, board_offset_x, board_offset_y)

    def draw_game_over(self, screen):
        font_size = max(20, int(min(screen.get_width(), screen.get_height()) // 10))
        message = f"Game Over! Score: {self.score} (R to Restart)"
        font = pygame.font.SysFont("arial", font_size)
        text_surface = font.render(message, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=screen.get_rect().center)

        # Shrink font size if too wide
        while text_rect.width > screen.get_width() * 0.95 and font_size > 10:
            font_size -= 2
            font = pygame.font.SysFont("arial", font_size)
            text_surface = font.render(message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=screen.get_rect().center)

        screen.blit(text_surface, text_rect)
    
    def draw_scoreboard(self, screen, board_offset_x=0, board_offset_y=0):
        font = pygame.font.SysFont("arial", 24)
        score_text = f"Score: {self.score}"
        text_surface = font.render(score_text, True, (255, 255, 255))
        # Place above the grid, aligned with grid edge
        screen.blit(text_surface, (board_offset_x, board_offset_y - 35))

    def reset(self):
        self.__init__(self.grid_size, self.cell_size, self.mode,
                      reward_fruit=self.reward_fruit,
                      reward_death=self.reward_death,
                      reward_step=self.reward_step)
        

    def get_state(self, device):
        state = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32, device=device)
        for (x, y) in self.snake.body:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                state[x, y] = 1.0
        fx, fy = self.fruit.position
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            state[fx, fy] = 2.0
        return state
    
    ACTIONS = [
        (0, -1),  # Up
        (0, 1),   # Down
        (-1, 0),  # Left
        (1, 0),   # Right
    ]

    def step(self, action_idx: int, device):
        """
        Apply action, update game state, and return (next_state, reward, done).
        Action index maps to direction as per ACTIONS list.
        """
        # Validate action index
        if action_idx not in range(len(self.ACTIONS)):
            raise ValueError(f"Invalid action index: {action_idx}")

        # Set snake direction
        self.snake.set_direction(self.ACTIONS[action_idx])

        # Save previous score to detect fruit eaten
        prev_score = self.score

        # Move snake and update game
        self.update()

        # Compute reward
        if not self.running:
            reward = self.reward_death
            done = True
        elif self.score > prev_score:
            reward = self.reward_fruit
            done = False
        else:
            reward = self.reward_step
            done = False

        # Get next observation
        next_state = self.get_state(device)

        return next_state, reward, done



	


