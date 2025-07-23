import random

def random_position(cell_count):
    """Generate a random (x, y) inside the grid."""
    x = random.randint(0, cell_count-1)
    y = random.randint(0, cell_count-1)
    return (x, y)
