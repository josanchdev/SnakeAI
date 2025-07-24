import pytest
from agent.memory import ReplayMemory

def test_memory_initialization():
    """Memory initializes empty with correct max size."""
    memory = ReplayMemory(max_size=5)
    assert len(memory) == 0
    assert memory.max_size == 5


def test_add_and_length():
    """Adding items updates memory size correctly."""
    memory = ReplayMemory(max_size=3)
    memory.add((1, 2, 3, 4, False))
    memory.add((2, 3, 4, 5, True))
    assert len(memory) == 2


def test_sample_batch():
    """Sampling returns requested number of experiences all present in memory."""
    memory = ReplayMemory(max_size=10)
    for i in range(6):
        memory.add((i, i+1, i+2, i+3, False))
    batch = memory.sample(4)
    assert len(batch) == 4
    for item in batch:
        assert item in memory.memory


def test_fifo_overflow():
    """Oldest memories get discarded when memory overflows."""
    memory = ReplayMemory(max_size=3)
    memory.add("first")
    memory.add("second")
    memory.add("third")
    memory.add("fourth")  # "first" should be dropped now
    assert len(memory) == 3
    assert "first" not in memory.memory
    assert "second" in memory.memory
    assert "third" in memory.memory
    assert "fourth" in memory.memory


def test_clear():
    """Clearing memory empties the buffer."""
    memory = ReplayMemory(max_size=3)
    memory.add(1)
    memory.add(2)
    memory.clear()
    assert len(memory) == 0


def test_is_full():
    """is_full returns True when buffer reaches max size."""
    memory = ReplayMemory(max_size=2)
    memory.add(1)
    assert not memory.is_full()
    memory.add(2)
    assert memory.is_full()


def test_sample_error_handling():
    """Sampling more than current size raises ValueError."""
    memory = ReplayMemory(max_size=2)
    memory.add(1)
    with pytest.raises(ValueError):
        memory.sample(2)