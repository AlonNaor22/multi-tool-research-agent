"""Calculator tool for the research agent.

A focused calculator with support for:
- Basic arithmetic and math functions
- Variables (store and reuse values)

Uses a safe evaluation approach to prevent code injection attacks.
"""

import math
import re
from typing import Dict, Optional
from langchain_core.tools import Tool


# ============================================================================
# SAFE MATH EVALUATION
# ============================================================================

# Allowed math functions (safe subset)
SAFE_MATH_FUNCTIONS = {
    # Basic
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,

    # From math module
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1/3),  # Cube root
    "exp": math.exp,
    "log": math.log,      # Natural log
    "log10": math.log10,  # Base 10 log
    "log2": math.log2,    # Base 2 log

    # Trigonometric (radians)
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,

    # Hyperbolic
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,

    # Angular conversion
    "degrees": math.degrees,  # radians to degrees
    "radians": math.radians,  # degrees to radians

    # Special functions
    "factorial": math.factorial,
    "gcd": math.gcd,
    "ceil": math.ceil,
    "floor": math.floor,
    "trunc": math.trunc,

    # Constants (as functions for easy access)
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "tau": lambda: math.tau,
    "inf": lambda: math.inf,
}


class AdvancedCalculator:
    """
    A calculator with variable storage and safe expression evaluation.

    Features:
    - Store and recall variables
    - Math functions (sqrt, sin, cos, factorial, etc.)
    - Safe evaluation (no code injection)
    """

    def __init__(self):
        """Initialize calculator with empty variable storage."""
        self.variables: Dict[str, float] = {}

    def set_variable(self, name: str, value: float) -> str:
        """Store a variable."""
        # Validate variable name (alphanumeric, starts with letter)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return f"Error: Invalid variable name '{name}'. Use letters, numbers, underscore (start with letter)."

        # Don't allow overwriting function names
        if name.lower() in SAFE_MATH_FUNCTIONS:
            return f"Error: '{name}' is a reserved function name."

        self.variables[name] = float(value)
        return f"Stored: {name} = {value}"

    def get_variable(self, name: str) -> Optional[float]:
        """Retrieve a variable value."""
        return self.variables.get(name)

    def list_variables(self) -> str:
        """List all stored variables."""
        if not self.variables:
            return "No variables stored."
        lines = ["Stored variables:"]
        for name, value in self.variables.items():
            lines.append(f"  {name} = {value}")
        return "\n".join(lines)

    def clear_variables(self) -> str:
        """Clear all stored variables."""
        self.variables.clear()
        return "All variables cleared."

    def safe_eval(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.

        This uses Python's eval() but with a restricted namespace
        containing only safe math functions and stored variables.
        """
        # Build the safe namespace
        safe_namespace = {}

        # Add math functions
        for name, func in SAFE_MATH_FUNCTIONS.items():
            # For constants like pi(), make them accessible as values
            if name in ("pi", "e", "tau", "inf"):
                safe_namespace[name] = func()
            else:
                safe_namespace[name] = func

        # Add stored variables
        safe_namespace.update(self.variables)

        # Validate expression doesn't contain dangerous patterns
        dangerous_patterns = [
            "__", "import", "exec", "eval", "compile", "open", "file",
            "input", "raw_input", "globals", "locals", "vars", "dir",
            "getattr", "setattr", "delattr", "hasattr",
            "type", "isinstance", "issubclass", "callable",
            "classmethod", "staticmethod", "property",
        ]

        expr_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                raise ValueError(f"Expression contains forbidden pattern: {pattern}")

        # Evaluate with restricted namespace
        # __builtins__ = {} prevents access to built-in functions
        result = eval(expression, {"__builtins__": {}}, safe_namespace)

        return float(result)

    def calculate(self, input_str: str) -> str:
        """
        Process a calculation request.

        Supports:
        - Simple expressions: "2 + 2"
        - Math functions: "sqrt(16)", "sin(pi/2)", "factorial(5)"
        - Variable assignment: "x = 10" or "set x = 10"
        - Variable usage: "x * 2" (after setting x)
        - List variables: "variables" or "vars"
        - Clear variables: "clear" or "clear variables"

        Args:
            input_str: The calculation request

        Returns:
            Result string or error message
        """
        input_str = input_str.strip()

        # Handle empty input
        if not input_str:
            return "Error: Empty expression"

        # Command: list variables
        if input_str.lower() in ("variables", "vars", "list"):
            return self.list_variables()

        # Command: clear variables
        if input_str.lower() in ("clear", "clear variables", "clear vars"):
            return self.clear_variables()

        # Command: help
        if input_str.lower() in ("help", "?"):
            return self._get_help()

        # Check for variable assignment: "x = 10" or "set x = 10"
        assignment_match = re.match(
            r'(?:set\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)',
            input_str
        )
        if assignment_match:
            var_name = assignment_match.group(1)
            expr = assignment_match.group(2)
            try:
                value = self.safe_eval(expr)
                return self.set_variable(var_name, value)
            except Exception as e:
                return f"Error evaluating expression: {str(e)}"

        # Otherwise, evaluate as expression
        try:
            result = self.safe_eval(input_str)

            # Format result nicely
            if result == int(result):
                return str(int(result))
            else:
                # Use general format, but limit decimal places
                return f"{result:.10g}"

        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: {str(e)}"
        except SyntaxError:
            return f"Error: Invalid syntax in expression"
        except NameError as e:
            return f"Error: Unknown variable or function - {str(e)}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def _get_help(self) -> str:
        """Return help text."""
        return """Calculator Help:

BASIC MATH:
  2 + 2, 10 * 5, 100 / 4, 2 ** 10, 17 % 5

MATH FUNCTIONS:
  sqrt(16), cbrt(27), abs(-5), round(3.7)
  sin(x), cos(x), tan(x), asin(x), acos(x), atan(x)
  log(x), log10(x), log2(x), exp(x)
  factorial(5), gcd(12, 8), ceil(3.2), floor(3.8)
  degrees(pi), radians(180)
  min(1,2,3), max(1,2,3), sum([1,2,3])

CONSTANTS:
  pi = 3.14159..., e = 2.71828..., tau = 6.28318...

VARIABLES:
  x = 10        (store a value)
  x * 2         (use stored value)
  variables     (list all variables)
  clear         (clear all variables)

For unit conversions, use the 'unit_converter' tool.
For solving equations, use the 'equation_solver' tool."""


# Create a global calculator instance (persists across calls)
_calculator = AdvancedCalculator()


def calculate(expression: str) -> str:
    """
    Entry point for the calculator tool.

    This function is called by the LangChain agent.
    """
    return _calculator.calculate(expression)


# Create the LangChain Tool wrapper
calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description=(
        "Perform mathematical calculations with variables. "
        "\n\nBASIC MATH: '2 + 2', '100 * 0.15', '2 ** 10' (power), '17 % 5' (modulo)"
        "\n\nMATH FUNCTIONS: sqrt(16), sin(x), cos(x), tan(x), log(x), log10(x), "
        "exp(x), factorial(5), abs(-5), round(3.7), ceil(x), floor(x), "
        "min(1,2,3), max(1,2,3), gcd(12,8), degrees(pi), radians(180)"
        "\n\nCONSTANTS: pi, e, tau"
        "\n\nVARIABLES: 'x = 10' to store, then 'x * 2' to use, 'variables' to list, 'clear' to reset"
        "\n\nNOTE: For unit conversions use 'unit_converter'. For solving equations use 'equation_solver'."
    )
)
