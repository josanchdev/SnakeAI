"""
Enhanced snake_game/game.py with better graphics and improved win detection
"""

import pygame
import sys
import numpy as np
import torch
from snake_game.utils import random_position

# Enhanced color scheme
COLORS = {
    'bg_dark': (26, 26, 32),
    'bg_light': (45, 45, 55),
    'border': (80, 80, 100),
    'snake_head': (0, 220, 60),
    'snake_body': (0, 180, 45),
    'snake_tail': (0, 140, 35),
    'fruit': (220, 60, 60),
    'fruit_glow': (255, 100, 100),
    'text_primary': (255, 255, 255),
    'text_secondary': (180, 180, 180),
    'win_text': (0, 255, 100),
    'game_over_text': (255, 100, 100)
}

class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        center = grid_size // 2
        self.body = [(center, center), (center-1, center), (center-2, center)]
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
        # Prevent reversing direction
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
        self.pulse = 0  # For glow animation

    def new_position(self, snake_body):
        # More efficient fruit placement
        available_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in snake_body:
                    available_positions.append((x, y))
        
        if not available_positions:
            # This shouldn't happen in normal gameplay
            return random_position(self.grid_size)
        
        import random
        return random.choice(available_positions)

    def respawn(self, snake_body):
        self.position = self.new_position(snake_body)
        self.pulse = 0

    def update_animation(self, dt):
        """Update fruit glow animation."""
        self.pulse += dt * 0.003  # Adjust speed as needed

class SnakeGame:
    def __init__(self, grid_size=12, cell_size=32, mode="human",
                 reward_fruit=10.0, reward_death=-10.0, reward_step=-0.01, reward_win=200.0):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.snake = Snake(grid_size)
        self.fruit = Fruit(grid_size, self.snake.body)
        self.score = 0
        self.running = True
        self.won = False
        self.mode = mode
        
        # Reward structure
        self.reward_fruit = reward_fruit
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_win = reward_win
        
        # Animation state
        self.time_elapsed = 0

    def check_win_condition(self):
        """Check if the snake has filled the entire grid (win condition)."""
        total_cells = self.grid_size * self.grid_size
        current_length = len(self.snake.body)
        
        # Account for fruit that will be eaten
        if self.snake.grow:
            current_length += 1
            
        # Win if snake fills the entire grid
        return current_length >= total_cells

    def ai_step(self, action_idx, device):
        """Step using AI action index."""
        next_state, reward, done = self.step(action_idx, device)
        return next_state, reward, done

    def update(self):
        """Update game state."""
        if not self.running:
            return
            
        self.snake.move()
        
        # Check fruit collision
        if self.snake.head() == self.fruit.position:
            self.snake.grow_snake()
            self.score += 1
            
            # Check win condition after eating fruit
            if self.check_win_condition():
                self.won = True
                self.running = False
                return
            
            # Respawn fruit if game continues
            self.fruit.respawn(self.snake.body)
            
        # Check collisions
        if self.snake.collided_with_self() or self.snake.collided_with_wall():
            self.running = False

    def draw(self, screen, board_offset_x=0, board_offset_y=0):
        """Draw the game with enhanced graphics."""
        # Update animation time
        self.time_elapsed += 16  # Assume ~60fps
        self.fruit.update_animation(self.time_elapsed)
        
        # Gradient background
        screen_rect = screen.get_rect()
        for y in range(screen_rect.height):
            color_ratio = y / screen_rect.height
            r = int(COLORS['bg_dark'][0] + (COLORS['bg_light'][0] - COLORS['bg_dark'][0]) * color_ratio)
            g = int(COLORS['bg_dark'][1] + (COLORS['bg_light'][1] - COLORS['bg_dark'][1]) * color_ratio)
            b = int(COLORS['bg_dark'][2] + (COLORS['bg_light'][2] - COLORS['bg_dark'][2]) * color_ratio)
            pygame.draw.line(screen, (r, g, b), (0, y), (screen_rect.width, y))

        # Draw grid border with rounded corners
        grid_w = self.grid_size * self.cell_size
        grid_h = self.grid_size * self.cell_size
        border_rect = pygame.Rect(board_offset_x - 2, board_offset_y - 2, grid_w + 4, grid_h + 4)
        pygame.draw.rect(screen, COLORS['border'], border_rect, width=0, border_radius=15)
        
        # Inner border
        inner_rect = pygame.Rect(board_offset_x, board_offset_y, grid_w, grid_h)
        pygame.draw.rect(screen, COLORS['bg_dark'], inner_rect, width=0, border_radius=12)

        # Draw snake with gradient effect
        for i, segment in enumerate(self.snake.body):
            x = board_offset_x + segment[0] * self.cell_size
            y = board_offset_y + segment[1] * self.cell_size
            
            if i == 0:  # Head
                color = COLORS['snake_head']
                # Add eyes to head
                pygame.draw.rect(screen, color, (x, y, self.cell_size, self.cell_size), border_radius=8)
                
                # Draw eyes
                eye_size = max(2, self.cell_size // 8)
                if self.snake.direction == (1, 0):  # Right
                    eye1_pos = (x + self.cell_size - eye_size * 2, y + eye_size)
                    eye2_pos = (x + self.cell_size - eye_size * 2, y + self.cell_size - eye_size * 2)
                elif self.snake.direction == (-1, 0):  # Left
                    eye1_pos = (x + eye_size, y + eye_size)
                    eye2_pos = (x + eye_size, y + self.cell_size - eye_size * 2)
                elif self.snake.direction == (0, -1):  # Up
                    eye1_pos = (x + eye_size, y + eye_size)
                    eye2_pos = (x + self.cell_size - eye_size * 2, y + eye_size)
                else:  # Down
                    eye1_pos = (x + eye_size, y + self.cell_size - eye_size * 2)
                    eye2_pos = (x + self.cell_size - eye_size * 2, y + self.cell_size - eye_size * 2)
                
                pygame.draw.circle(screen, (255, 255, 255), eye1_pos, eye_size)
                pygame.draw.circle(screen, (255, 255, 255), eye2_pos, eye_size)
                pygame.draw.circle(screen, (0, 0, 0), eye1_pos, max(1, eye_size // 2))
                pygame.draw.circle(screen, (0, 0, 0), eye2_pos, max(1, eye_size // 2))
                
            elif i == len(self.snake.body) - 1:  # Tail
                color = COLORS['snake_tail']
                pygame.draw.rect(screen, color, (x, y, self.cell_size, self.cell_size), border_radius=6)
            else:  # Body
                color = COLORS['snake_body']
                pygame.draw.rect(screen, color, (x, y, self.cell_size, self.cell_size), border_radius=7)
        
        # Draw animated fruit (only if not won)
        if not self.won:
            fx, fy = self.fruit.position
            x = board_offset_x + fx * self.cell_size
            y = board_offset_y + fy * self.cell_size
            
            # Pulsing glow effect
            glow_intensity = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(self.fruit.pulse))
            glow_color = tuple(int(COLORS['fruit_glow'][i] * glow_intensity) for i in range(3))
            
            # Draw glow
            glow_size = int(self.cell_size * (1.2 + 0.1 * np.sin(self.fruit.pulse)))
            glow_rect = pygame.Rect(
                x - (glow_size - self.cell_size) // 2,
                y - (glow_size - self.cell_size) // 2,
                glow_size, glow_size
            )
            pygame.draw.ellipse(screen, glow_color, glow_rect)
            
            # Draw fruit
            fruit_rect = pygame.Rect(x + 2, y + 2, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.ellipse(screen, COLORS['fruit'], fruit_rect)
            
            # Add shine effect
            shine_rect = pygame.Rect(x + 4, y + 4, self.cell_size // 3, self.cell_size // 3)
            pygame.draw.ellipse(screen, (255, 200, 200), shine_rect)
        
        # Draw score and status
        self.draw_scoreboard(screen, board_offset_x, board_offset_y)

    def draw_game_over(self, screen):
        """Draw game over screen with better styling."""
        # Semi-transparent overlay
        overlay = pygame.Surface(screen.get_size())
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))
        
        # Game over message
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 32)
        
        if self.won:
            title = "🎉 VICTORY! 🎉"
            message = f"Perfect Score: {self.score}"
            color = COLORS['win_text']
        else:
            title = "💀 Game Over 💀"
            message = f"Final Score: {self.score}"
            color = COLORS['game_over_text']
        
        # Draw title
        title_surface = font_large.render(title, True, color)
        title_rect = title_surface.get_rect(center=(screen.get_width()//2, screen.get_height()//2 - 40))
        screen.blit(title_surface, title_rect)
        
        # Draw score
        score_surface = font_medium.render(message, True, COLORS['text_primary'])
        score_rect = score_surface.get_rect(center=(screen.get_width()//2, screen.get_height()//2 + 10))
        screen.blit(score_surface, score_rect)
        
        # Draw restart instruction
        restart_surface = font_medium.render("Press R to restart", True, COLORS['text_secondary'])
        restart_rect = restart_surface.get_rect(center=(screen.get_width()//2, screen.get_height()//2 + 50))
        screen.blit(restart_surface, restart_rect)
    
    def draw_scoreboard(self, screen, board_offset_x=0, board_offset_y=0):
        """Draw enhanced scoreboard."""
        font = pygame.font.Font(None, 28)
        
        # Score text
        score_text = f"Score: {self.score}"
        if self.won:
            score_text += " 🏆 PERFECT!"
            color = COLORS['win_text']
        else:
            color = COLORS['text_primary']
        
        text_surface = font.render(score_text, True, color)
        
        # Background for score
        text_rect = text_surface.get_rect()
        bg_rect = pygame.Rect(board_offset_x, board_offset_y - 45, text_rect.width + 20, 35)
        pygame.draw.rect(screen, (*COLORS['bg_dark'], 200), bg_rect, border_radius=8)
        
        # Draw score text
        screen.blit(text_surface, (board_offset_x + 10, board_offset_y - 40))

    def reset(self):
        """Reset the game."""
        self.__init__(
            self.grid_size, self.cell_size, self.mode,
            reward_fruit=self.reward_fruit,
            reward_death=self.reward_death,
            reward_step=self.reward_step,
            reward_win=self.reward_win
        )

    def get_state(self, device):
        """Get game state as tensor."""
        # Grid encoding (144D for 12x12 grid)
        state = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32, device=device)
        
        # Encode snake body
        for (x, y) in self.snake.body:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                state[x, y] = 1.0
        
        # Encode fruit
        fx, fy = self.fruit.position
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            state[fx, fy] = 2.0

        # Direction one-hot encoding (4D)
        dir_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        direction = torch.zeros(4, device=device)
        direction[dir_map.get(self.snake.direction, 0)] = 1.0

        # Relative fruit position (2D, normalized)
        head_x, head_y = self.snake.head()
        dx = (fx - head_x) / (self.grid_size - 1)
        dy = (fy - head_y) / (self.grid_size - 1)
        rel_fruit = torch.tensor([dx, dy], dtype=torch.float32, device=device)

        # Combine all features: 144 + 4 + 2 = 150D
        flat_grid = state.flatten()
        full_state = torch.cat([flat_grid, direction, rel_fruit])
        return full_state

    def step(self, action_idx: int, device):
        """Execute one game step with given action."""
        from agent.dqn import ACTIONS
        
        if not isinstance(action_idx, int) or not (0 <= action_idx < len(ACTIONS)):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        # Apply action
        self.snake.set_direction(ACTIONS[action_idx])
        prev_score = self.score
        
        # Update game
        self.update()
        
        # Calculate reward
        if not self.running:
            if self.won:
                reward = self.reward_win  # Perfect game bonus
            else:
                reward = self.reward_death  # Death penalty
            done = True
        elif self.score > prev_score:
            reward = self.reward_fruit  # Fruit eaten
            done = False
        else:
            reward = self.reward_step  # Time penalty (encourages efficiency)
            done = False
            
        next_state = self.get_state(device)
        return next_state, reward, done