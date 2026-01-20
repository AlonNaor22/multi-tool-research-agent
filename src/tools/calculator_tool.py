"""Calculator tool for the research agent.

Uses numexpr for safe mathematical expression evaluation.
This prevents code injection attacks that would be possible with Python's eval().
"""

import numexpr as ne
from langchain_core.tools import Tool


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 2" or "100 * 0.15"

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        # numexpr.evaluate() safely computes math expressions
        # It cannot execute arbitrary Python code, only math
        result = ne.evaluate(expression)

        # .item() converts numpy scalar to Python native type
        return str(result.item())
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# Create the LangChain Tool wrapper
# This is what we'll give to the agent
calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description=(
        "Perform mathematical calculations. Use this when you need to do "
        "arithmetic, percentages, or any math computation. "
        "Input should be a mathematical expression (e.g., '2 + 2', '100 * 0.15', "
        "'(50 - 30) / 20'). Supports +, -, *, /, ** (power), and parentheses."
    )
)
