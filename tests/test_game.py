import pytest
import numpy as np
from snake_game.game import SnakeGame

def test_step_eating_fruit_and_death():
    game = SnakeGame(grid_size=5)  # smaller grid for simplicity

    # Place fruit directly in front of snake head to ensure eating
    head_x, head_y = game.snake.head()
    fruit_pos = (head_x + 1, head_y)  # assuming initial direction right (1,0)
    game.fruit.position = fruit_pos

    # Take action 'right' (index 3)
    next_state, reward, done = game.step(3)
    assert isinstance(next_state, np.ndarray)
    assert reward == 1  # fruit eaten
    assert not done

    # Move snake into wall to simulate death
    game.snake.body = [(game.grid_size-1, game.grid_size-1)]
    game.snake.direction = (1, 0)  # moving right into wall
    next_state, reward, done = game.step(3)
    assert reward == -10  # death penalty
    assert done

def test_invalid_action_index():
    game = SnakeGame()
    with pytest.raises(ValueError):
        game.step(10)  # invalid action index
