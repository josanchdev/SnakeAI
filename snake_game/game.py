import pygame
import sys
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
    def __init__(self, grid_size=12, cell_size=32):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.snake = Snake(grid_size)
        self.fruit = Fruit(grid_size, self.snake.body)
        self.score = 0
        self.running = True

    def update(self):
        self.snake.move()
        if self.snake.head() == self.fruit.position:
            self.snake.grow_snake()
            self.fruit.respawn(self.snake.body)
            self.score += 1
        if self.snake.collided_with_self() or self.snake.collided_with_wall():
            self.running = False

    def draw(self, screen):
        screen.fill((0, 0, 0))
        for segment in self.snake.body:
            pygame.draw.rect(
                screen, (0, 180, 45),
                (segment[0]*self.cell_size, segment[1]*self.cell_size, self.cell_size, self.cell_size)
            )
        fx, fy = self.fruit.position
        pygame.draw.rect(
            screen, (200, 60, 60),
            (fx*self.cell_size, fy*self.cell_size, self.cell_size, self.cell_size)
        )
        self.draw_scoreboard(screen)

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
    
    def draw_scoreboard(self, screen):
        font = pygame.font.SysFont("arial", 24)
        score_text = f"Score: {self.score}"
        text_surface = font.render(score_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

    def reset(self):
        self.__init__(self.grid_size, self.cell_size)

if __name__ == "__main__":
    pygame.init()
    grid_size = 15
    cell_size = 32
    screen = pygame.display.set_mode((grid_size*cell_size, grid_size*cell_size))
    clock = pygame.time.Clock()
    pygame.display.set_caption("AI-Ready Minimal Snake - Pygame")

    game = SnakeGame(grid_size, cell_size)
    MOVE_EVENT = pygame.USEREVENT
    pygame.time.set_timer(MOVE_EVENT, 90)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_UP:
                    game.snake.set_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    game.snake.set_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    game.snake.set_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    game.snake.set_direction((1, 0))
                elif event.key == pygame.K_r and not game.running:
                    game.reset()
            if event.type == MOVE_EVENT and game.running:
                game.update()

        game.draw(screen)
        if not game.running:
            game.draw_game_over(screen)
        pygame.display.flip()
        clock.tick(60)
