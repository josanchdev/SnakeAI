"""
Enhanced main.py with beautiful menu and improved game flow
"""

import pygame
import os
import torch
import numpy as np
from snake_game.game import SnakeGame
from agent.dqn import DQN
from config import GRID_SIZE, CELL_SIZE, ModelConfig

# Color scheme
COLORS = {
    'bg_dark': (26, 26, 32),
    'bg_light': (45, 45, 55),
    'accent': (0, 180, 45),
    'text_primary': (255, 255, 255),
    'text_secondary': (180, 180, 180),
    'button': (70, 70, 85),
    'button_hover': (90, 90, 105),
    'snake': (0, 180, 45),
    'fruit': (220, 60, 60)
}

class GameMenu:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Menu state
        self.selected_option = 0
        self.options = [
            ("🎮 Play Human vs Snake", "human"),
            ("🤖 Watch AI Play", "ai"),
            ("📊 AI vs Human Demo", "demo"),
            ("❌ Quit", "quit")
        ]
        
        # Button rectangles
        self.button_rects = []
        self._create_buttons()
    
    def _create_buttons(self):
        """Create button rectangles for menu options."""
        button_width = 300
        button_height = 60
        button_spacing = 20
        
        start_y = self.height // 2 - (len(self.options) * (button_height + button_spacing)) // 2
        
        for i, (text, _) in enumerate(self.options):
            x = self.width // 2 - button_width // 2
            y = start_y + i * (button_height + button_spacing)
            self.button_rects.append(pygame.Rect(x, y, button_width, button_height))
    
    def handle_event(self, event):
        """Handle menu events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self.options[self.selected_option][1]
        
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            for i, rect in enumerate(self.button_rects):
                if rect.collidepoint(mouse_pos):
                    self.selected_option = i
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                return self.options[self.selected_option][1]
        
        return None
    
    def draw(self):
        """Draw the main menu."""
        # Gradient background
        for y in range(self.height):
            color_ratio = y / self.height
            r = int(COLORS['bg_dark'][0] + (COLORS['bg_light'][0] - COLORS['bg_dark'][0]) * color_ratio)
            g = int(COLORS['bg_dark'][1] + (COLORS['bg_light'][1] - COLORS['bg_dark'][1]) * color_ratio)
            b = int(COLORS['bg_dark'][2] + (COLORS['bg_light'][2] - COLORS['bg_dark'][2]) * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
        
        # Title
        title = self.font_large.render("🐍 SlytherNN", True, COLORS['accent'])
        subtitle = self.font_medium.render("Deep Reinforcement Learning Snake", True, COLORS['text_secondary'])
        
        title_rect = title.get_rect(center=(self.width // 2, 100))
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 140))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu buttons
        for i, ((text, _), rect) in enumerate(zip(self.options, self.button_rects)):
            # Button background
            is_selected = i == self.selected_option
            button_color = COLORS['button_hover'] if is_selected else COLORS['button']
            
            pygame.draw.rect(self.screen, button_color, rect, border_radius=10)
            
            if is_selected:
                pygame.draw.rect(self.screen, COLORS['accent'], rect, width=3, border_radius=10)
            
            # Button text
            text_surface = self.font_medium.render(text, True, COLORS['text_primary'])
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        # Instructions
        instructions = [
            "Use ↑↓ keys or mouse to navigate",
            "Press ENTER/SPACE or click to select",
            "ESC to quit anytime"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS['text_secondary'])
            text_rect = text.get_rect(center=(self.width // 2, self.height - 80 + i * 20))
            self.screen.blit(text, text_rect)


class GameSession:
    def __init__(self, screen, mode):
        self.screen = screen
        self.mode = mode
        self.width, self.height = screen.get_size()
        
        # Calculate board positioning
        grid_pixel_size = GRID_SIZE * CELL_SIZE
        self.board_offset_x = (self.width - grid_pixel_size) // 2
        self.board_offset_y = (self.height - grid_pixel_size) // 2
        
        self.game = SnakeGame(GRID_SIZE, CELL_SIZE, mode=mode)
        self.ai_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load AI model if needed
        if mode in ["ai", "demo"]:
            self._load_ai_model()
        
        # Game timing
        self.move_timer = 0
        self.move_delay = 150 if mode == "ai" else 120  # ms between moves
        
        # Demo mode state
        self.demo_human_turn = mode == "demo"
        self.demo_switch_timer = 0
        self.demo_switch_delay = 5000  # 5 seconds per turn
    
    def _load_ai_model(self):
        """Load the trained AI model."""
        checkpoint_path = self._get_latest_checkpoint()
        if checkpoint_path:
            try:
                self.ai_model = DQN(
                    input_dim=ModelConfig.INPUT_DIM,
                    output_dim=ModelConfig.OUTPUT_DIM,
                    hidden_dims=ModelConfig.HIDDEN_DIMS
                ).to(self.device)
                
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model_state = state['model'] if isinstance(state, dict) and 'model' in state else state
                self.ai_model.load_state_dict(model_state)
                self.ai_model.eval()
                
                print(f"✅ AI model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load AI model: {e}")
                self.ai_model = None
        else:
            print("⚠️ No trained model found. Train a model first with train.py")
    
    def _get_latest_checkpoint(self):
        """Get the latest checkpoint file."""
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            return None
        
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not files:
            return None
        
        files.sort(key=lambda x: int(x.split('_ep')[1].split('.pth')[0]))
        return os.path.join(checkpoint_dir, files[-1])
    
    def handle_event(self, event):
        """Handle game events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "menu"
            elif event.key == pygame.K_r and not self.game.running:
                self._reset_game()
            elif self.mode == "human" or (self.mode == "demo" and self.demo_human_turn):
                self._handle_human_input(event)
        
        return None
    
    def _handle_human_input(self, event):
        """Handle human player input."""
        direction_map = {
            pygame.K_UP: (0, -1),
            pygame.K_DOWN: (0, 1),
            pygame.K_LEFT: (-1, 0),
            pygame.K_RIGHT: (1, 0)
        }
        
        if event.key in direction_map:
            self.game.snake.set_direction(direction_map[event.key])
    
    def update(self, dt):
        """Update game state."""
        if not self.game.running:
            return
        
        # Handle demo mode switching
        if self.mode == "demo":
            self.demo_switch_timer += dt
            if self.demo_switch_timer >= self.demo_switch_delay:
                self.demo_human_turn = not self.demo_human_turn
                self.demo_switch_timer = 0
                self._reset_game()
        
        # Game movement timing
        self.move_timer += dt
        if self.move_timer >= self.move_delay:
            self.move_timer = 0
            
            if self.mode == "ai" or (self.mode == "demo" and not self.demo_human_turn):
                self._ai_move()
            else:
                self.game.update()
    
    def _ai_move(self):
        """Execute AI move."""
        if self.ai_model is None:
            self.game.update()
            return
        
        try:
            state = self.game.get_state(self.device).flatten()
            state_tensor = state.unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.ai_model(state_tensor)
                action_idx = torch.argmax(q_values).item()
            
            self.game.ai_step(action_idx, self.device)
        except Exception as e:
            print(f"AI move error: {e}")
            self.game.update()
    
    def _reset_game(self):
        """Reset the game."""
        current_mode = "human" if (self.mode == "demo" and self.demo_human_turn) else self.mode
        self.game = SnakeGame(GRID_SIZE, CELL_SIZE, mode=current_mode)
    
    def draw(self):
        """Draw the game."""
        # Background
        self.screen.fill(COLORS['bg_dark'])
        
        # Game area
        self.game.draw(self.screen, self.board_offset_x, self.board_offset_y)
        
        # UI overlays
        self._draw_ui()
        
        if not self.game.running:
            self.game.draw_game_over(self.screen)
    
    def _draw_ui(self):
        """Draw UI elements."""
        font = pygame.font.Font(None, 24)
        
        # Mode indicator
        if self.mode == "demo":
            current_player = "HUMAN" if self.demo_human_turn else "AI"
            time_left = (self.demo_switch_delay - self.demo_switch_timer) / 1000
            mode_text = f"Demo Mode - {current_player} playing ({time_left:.1f}s)"
            color = COLORS['accent'] if self.demo_human_turn else COLORS['fruit']
        else:
            mode_text = f"Mode: {'Human' if self.mode == 'human' else 'AI'}"
            color = COLORS['text_primary']
        
        text_surface = font.render(mode_text, True, color)
        self.screen.blit(text_surface, (10, 10))
        
        # Controls
        if self.mode == "human" or (self.mode == "demo" and self.demo_human_turn):
            controls = "Controls: ↑↓←→ to move, R to restart, ESC for menu"
        else:
            controls = "Watching AI play - R to restart, ESC for menu"
        
        controls_surface = font.render(controls, True, COLORS['text_secondary'])
        self.screen.blit(controls_surface, (10, self.height - 30))


def main():
    """Main game loop."""
    pygame.init()
    
    # Screen setup
    screen_width, screen_height = 800, 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("🐍 SlytherNN - Deep RL Snake")
    
    clock = pygame.time.Clock()
    
    # Game state
    state = "menu"  # "menu", "game"
    menu = GameMenu(screen)
    game_session = None
    
    running = True
    while running:
        dt = clock.tick(60)  # 60 FPS
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if state == "game":
                    state = "menu"
                else:
                    running = False
            
            if state == "menu":
                action = menu.handle_event(event)
                if action:
                    if action == "quit":
                        running = False
                    else:
                        state = "game"
                        game_session = GameSession(screen, action)
            
            elif state == "game" and game_session:
                result = game_session.handle_event(event)
                if result == "menu":
                    state = "menu"
        
        # Update
        if state == "game" and game_session:
            game_session.update(dt)
        
        # Draw
        if state == "menu":
            menu.draw()
        elif state == "game" and game_session:
            game_session.draw()
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    main()