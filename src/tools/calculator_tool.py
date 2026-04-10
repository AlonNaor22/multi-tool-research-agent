"""Calculator tool — safe math evaluation with variables and step-by-step solutions."""

import json
import math
import re
from typing import Dict, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.tools.step_solver import detect_operation, StepByStepSolver

# Global step solver instance
_step_solver = StepByStepSolver()


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
    """Calculator with variable storage and safe expression evaluation."""

    def __init__(self):
        """Initialize with empty variable storage."""
        self.variables: Dict[str, float] = {}

    def set_variable(self, name: str, value: float) -> str:
        """Store a named variable for later use in expressions."""
        # Validate variable name (alphanumeric, starts with letter)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return f"Error: Invalid variable name '{name}'. Use letters, numbers, underscore (start with letter)."

        # Don't allow overwriting function names
        if name.lower() in SAFE_MATH_FUNCTIONS:
            return f"Error: '{name}' is a reserved function name."

        self.variables[name] = float(value)
        return f"Stored: {name} = {value}"

    def get_variable(self, name: str) -> Optional[float]:
        """Retrieve a stored variable value, or None."""
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
        """Evaluate a math expression in a restricted namespace (no builtins)."""
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
        """Process a calculation, variable assignment, or step-by-step request."""
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

        # Check for step-by-step operations (calculus, matrix, complex arithmetic)
        operation, cleaned = detect_operation(input_str)
        if operation not in ("simple", "passthrough"):
            # Return structured JSON for the math_formatter tool to render
            structured = _step_solver.solve_structured(operation, cleaned)
            if structured.get("error") and not structured.get("steps"):
                return f"Error: {structured['error']}"
            # Also include the plain-text version as fallback
            plain_text = _step_solver.solve(operation, cleaned)
            structured["plain_text"] = plain_text
            return "MATH_STRUCTURED:" + json.dumps(structured, default=str)

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
        return """Calculator Help (with step-by-step solutions):

BASIC MATH:
  2 + 2, 10 * 5, 100 / 4, 2 ** 10, 17 % 5

STEP-BY-STEP (automatically for complex expressions):
  (2+3)*4 - 10/2    -> shows order of operations
  derivative of x^3 + 2x   -> shows power rule steps
  integrate x^2 from 0 to 5 -> shows antiderivative + bounds
  solve x^2 - 4 = 0  -> shows discriminant + quadratic formula

MATRIX OPERATIONS (with steps):
  determinant [[1,2],[3,4]]
  inverse [[1,2],[3,4]]
  [[1,2],[3,4]] * [[5,6],[7,8]]
  transpose [[1,2,3],[4,5,6]]
  [[1,2],[3,4]] + [[5,6],[7,8]]

MATH FUNCTIONS:
  sqrt(16), cbrt(27), abs(-5), round(3.7)
  sin(x), cos(x), tan(x), log(x), exp(x)
  factorial(5), gcd(12, 8), ceil(3.2), floor(3.8)

CONSTANTS:
  pi = 3.14159..., e = 2.71828..., tau = 6.28318...

VARIABLES:
  x = 10        (store a value)
  x * 2         (use stored value)
  variables     (list all variables)
  clear         (clear all variables)

For unit conversions, use the 'unit_converter' tool.
For symbolic algebra (simplify/expand/factor), systems, eigenvalues, use 'equation_solver'."""


# Create a global calculator instance (persists across calls)
_calculator = AdvancedCalculator()


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression, equation, variable assignment, or command (help/vars/clear)")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "Perform mathematical calculations with step-by-step solutions for students. "
        "\n\nBASIC MATH: '2 + 2', '100 * 0.15', '2 ** 10' (power), '17 % 5' (modulo)"
        "\n\nSTEP-BY-STEP: Complex expressions automatically show solution steps."
        "\n\nCALCULUS: 'derivative of x^3 + 2x', 'integrate x^2 from 0 to 5'"
        "\n\nMATRIX OPS: 'determinant [[1,2],[3,4]]', 'inverse [[1,2],[3,4]]', "
        "'[[1,2],[3,4]] * [[5,6],[7,8]]', 'transpose [[1,2,3],[4,5,6]]', "
        "'[[1,2],[3,4]] + [[5,6],[7,8]]'"
        "\n\nEQUATIONS: 'solve x^2 - 4 = 0' (with solution steps)"
        "\n\nMATH FUNCTIONS: sqrt(16), sin(x), cos(x), factorial(5), abs(-5), ceil(x), floor(x)"
        "\n\nCONSTANTS: pi, e, tau"
        "\n\nVARIABLES: 'x = 10' to store, then 'x * 2' to use, 'variables' to list, 'clear' to reset"
        "\n\nNOTE: For unit conversions use 'unit_converter'. For symbolic algebra "
        "(simplify/expand/factor), systems of equations, eigenvalues, or RREF use 'equation_solver'."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str = "") -> str:
        return _calculator.calculate(expression)

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


calculator_tool = CalculatorTool()

# Public function for direct calls (tests, CLI)
calculate = _calculator.calculate
