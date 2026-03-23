"""Tests for SimpleMemory class in src/agent.py.

We import SimpleMemory carefully to avoid pulling in langchain dependencies
that may not be installed in the test environment.
"""

import sys
import types
import pytest


def _import_simple_memory():
    """Import SimpleMemory without triggering full agent.py imports."""
    import importlib
    import ast

    # Read the source and extract just the SimpleMemory class
    agent_path = "src/agent.py"
    with open(agent_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    # Find the SimpleMemory class definition
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SimpleMemory":
            # Extract lines for the class
            start = node.lineno - 1
            end = node.end_lineno
            class_source = "\n".join(source.split("\n")[start:end])

            # Execute the class definition in isolation
            namespace = {}
            exec(class_source, namespace)
            return namespace["SimpleMemory"]

    raise ImportError("Could not find SimpleMemory class")


SimpleMemory = _import_simple_memory()


@pytest.fixture
def memory():
    """Fresh memory instance with default window size."""
    return SimpleMemory(k=5)


class TestMemoryBasics:
    """Test basic memory operations."""

    def test_empty_memory(self, memory):
        assert memory.get_history_string() == "No previous conversation."

    def test_add_exchange(self, memory):
        memory.add_exchange("What is AI?", "AI is artificial intelligence.")
        assert len(memory.history) == 1

    def test_history_grows(self, memory):
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.add_exchange("Q3", "A3")
        assert len(memory.history) == 3

    def test_history_string_contains_exchanges(self, memory):
        memory.add_exchange("What is Python?", "A programming language.")
        result = memory.get_history_string()
        assert "What is Python?" in result
        assert "A programming language." in result


class TestMemoryWindow:
    """Test that only last k exchanges are sent to prompt."""

    def test_window_limits_prompt(self):
        memory = SimpleMemory(k=2)
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.add_exchange("Q3", "A3")

        result = memory.get_history_string()
        # Should only contain last 2 exchanges
        assert "Q1" not in result
        assert "Q2" in result
        assert "Q3" in result

    def test_full_history_preserved(self):
        memory = SimpleMemory(k=2)
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.add_exchange("Q3", "A3")

        # All exchanges should still be in the full history
        assert len(memory.history) == 3


class TestMemoryClear:
    """Test clearing memory."""

    def test_clear_empties_history(self, memory):
        memory.add_exchange("Q1", "A1")
        memory.add_exchange("Q2", "A2")
        memory.clear()
        assert len(memory.history) == 0
        assert memory.get_history_string() == "No previous conversation."


class TestMemoryBuffer:
    """Test buffer property for compatibility."""

    def test_buffer_returns_history_string(self, memory):
        memory.add_exchange("Q1", "A1")
        assert memory.buffer == memory.get_history_string()
