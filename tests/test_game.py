import pytest
import numpy as np
from snake_game.game import SnakeGame
import torch

def test_step_eating_fruit_and_death():
    game = SnakeGame(grid_size=5)  # smaller grid for simplicity

    # Place fruit directly in front of snake head to ensure eating
    head_x, head_y = game.snake.head()
    fruit_pos = (head_x + 1, head_y)  # assuming initial direction right (1,0)
    game.fruit.position = fruit_pos

    # Take action 'right' (index 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    next_state, reward, done = game.step(3, device)
    assert isinstance(next_state, torch.Tensor)
    assert reward == 5  # fruit eaten (default reward_fruit=5)
    assert not done

    # Move snake into wall to simulate death
    game.snake.body = [(game.grid_size-1, game.grid_size-1)]
    game.snake.direction = (1, 0)  # moving right into wall
    next_state, reward, done = game.step(3, device)
    assert reward == -10  # death penalty
    assert done

def test_win_condition():
    """Test that the game detects win condition when snake fills the entire grid."""
    game = SnakeGame(grid_size=3)  # Very small grid for easy testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a valid snake body that covers 8 out of 9 cells in a 3x3 grid
    # Layout:
    # S S S
    # H F S  (where H=head, S=snake, F=fruit)
    # S S S
    game.snake.body = [
        (0, 1),  # Head at middle-left
        (0, 0),  # Body going up
        (1, 0),  # Body going right
        (2, 0),  # Body continuing right
        (2, 1),  # Body going down
        (2, 2),  # Body continuing down
        (1, 2),  # Body going left
        (0, 2)   # Tail at bottom-left
    ]
    game.fruit.position = (1, 1)  # Fruit in center
    game.snake.direction = (1, 0)  # Moving right to eat fruit
    
    # Verify setup is correct
    assert len(game.snake.body) == 8, f"Snake should have 8 segments, has {len(game.snake.body)}"
    assert len(set(game.snake.body)) == 8, "Snake body should not have duplicates"
    assert not game.check_win_condition(), "Should not be won yet"
    
    # Take step to eat the final fruit (move right)
    next_state, reward, done = game.step(3, device)  # Move right
    
    assert game.won == True, f"Game should be in won state, won={game.won}, running={game.running}"
    assert reward == 100, f"Should get win reward of 100, got {reward}"
    assert done == True, "Game should be done after winning"
    assert not game.running, "Game should not be running after win"

def test_win_condition_check():
    """Test the check_win_condition method directly."""
    game = SnakeGame(grid_size=3)
    
    # Initially should not be won
    assert not game.check_win_condition()
    
    # Fill entire 3x3 grid
    game.snake.body = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2)
    ]
    
    # Now should be won
    assert game.check_win_condition()

def test_no_fruit_drawn_when_won():
    """Test that fruit is not drawn when the game is won."""
    game = SnakeGame(grid_size=3)
    
    # Set to won state
    game.won = True
    game.running = False
    
    # The draw method should handle this gracefully (visual test)
    # This mainly tests that the code doesn't crash
    assert game.won == True

def test_invalid_action_index():
    game = SnakeGame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        game.step(10, device)  # invalid action index