"""Equation solver tool for the research agent.

A dedicated tool for solving mathematical equations with unknown variables.
Uses SymPy for symbolic mathematics.

Supports:
- Linear equations: x + 2 = 5
- Quadratic equations: x^2 - 4 = 0
- Equations with multiple variables: solve for one variable
- Systems of equations (basic support)
"""

import re
from typing import List, Optional
from langchain_core.tools import Tool

try:
    from sympy import symbols, Eq, solve, sympify, SympifyError
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


def _preprocess_equation(equation_str: str) -> str:
    """
    Preprocess equation string to make it SymPy-compatible.

    Converts:
    - x^2 to x**2 (power notation)
    - 2x to 2*x (implicit multiplication)
    """
    # Replace ^ with ** for powers
    equation_str = equation_str.replace("^", "**")

    # Add multiplication between number and variable (e.g., 2x -> 2*x)
    equation_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation_str)

    # Add multiplication between closing paren and variable (e.g., (2)x -> (2)*x)
    equation_str = re.sub(r'\)([a-zA-Z])', r')*\1', equation_str)

    # Add multiplication between variable and opening paren (e.g., x(2) -> x*(2))
    equation_str = re.sub(r'([a-zA-Z])\(', r'\1*(', equation_str)

    return equation_str


def _extract_variables(expression_str: str) -> List[str]:
    """Extract variable names from an expression string."""
    # Find all single letters that could be variables
    # Exclude common function names
    reserved = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs', 'pi', 'e'}

    # Find potential variables (single letters or letter followed by numbers)
    potential_vars = re.findall(r'\b([a-zA-Z])\b', expression_str)

    # Filter out reserved names and duplicates
    variables = []
    seen = set()
    for var in potential_vars:
        if var.lower() not in reserved and var not in seen:
            variables.append(var)
            seen.add(var)

    return variables


def solve_equation(input_str: str) -> str:
    """
    Solve an equation for unknown variable(s).

    Supports formats:
    - "x + 2 = 5"
    - "solve x + 2 = 5"
    - "x^2 - 4 = 0"
    - "2x + 3 = 11"
    - "solve for x: 2x + y = 10" (when y is known or solving for x in terms of y)

    Args:
        input_str: The equation to solve

    Returns:
        Solution string or error message
    """
    if not SYMPY_AVAILABLE:
        return "Error: SymPy library is not installed. Run: pip install sympy"

    input_str = input_str.strip()

    # Handle empty input
    if not input_str:
        return "Error: Empty equation"

    # Command: help
    if input_str.lower() in ("help", "?"):
        return _get_help()

    # Remove "solve" prefix if present
    if input_str.lower().startswith("solve"):
        input_str = input_str[5:].strip()

    # Remove "for x:" type prefix if present
    solve_for_match = re.match(r'for\s+([a-zA-Z])\s*:\s*(.+)', input_str, re.IGNORECASE)
    target_var = None
    if solve_for_match:
        target_var = solve_for_match.group(1)
        input_str = solve_for_match.group(2)

    # Check if there's an equals sign
    if "=" not in input_str:
        # Try to solve expression = 0
        input_str = input_str + " = 0"

    # Split by equals sign
    parts = input_str.split("=")
    if len(parts) != 2:
        return "Error: Equation must have exactly one '=' sign"

    left_side = parts[0].strip()
    right_side = parts[1].strip()

    # Preprocess both sides
    left_side = _preprocess_equation(left_side)
    right_side = _preprocess_equation(right_side)

    try:
        # Extract variables
        all_vars = _extract_variables(left_side + " " + right_side)

        if not all_vars:
            return "Error: No variables found in equation"

        # Create SymPy symbols
        sym_vars = {var: symbols(var) for var in all_vars}

        # Parse expressions
        transformations = standard_transformations + (implicit_multiplication_application,)

        left_expr = parse_expr(left_side, local_dict=sym_vars, transformations=transformations)
        right_expr = parse_expr(right_side, local_dict=sym_vars, transformations=transformations)

        # Create equation
        equation = Eq(left_expr, right_expr)

        # Determine which variable to solve for
        if target_var and target_var in sym_vars:
            solve_var = sym_vars[target_var]
        elif len(all_vars) == 1:
            solve_var = sym_vars[all_vars[0]]
        else:
            # Default to 'x' if present, otherwise first variable
            if 'x' in all_vars:
                solve_var = sym_vars['x']
            else:
                solve_var = sym_vars[all_vars[0]]

        # Solve the equation
        solutions = solve(equation, solve_var)

        # Format the result
        if not solutions:
            return f"No solution found for {solve_var}"

        var_name = str(solve_var)

        if len(solutions) == 1:
            sol = solutions[0]
            # Check if solution is numeric
            if sol.is_number:
                # Evaluate to get decimal if it's a complex expression
                numeric_val = float(sol.evalf())
                if numeric_val == int(numeric_val):
                    return f"{var_name} = {int(numeric_val)}"
                else:
                    return f"{var_name} = {numeric_val:.6g}"
            else:
                return f"{var_name} = {sol}"
        else:
            # Multiple solutions
            formatted_solutions = []
            for sol in solutions:
                if sol.is_number:
                    numeric_val = complex(sol.evalf())
                    if numeric_val.imag == 0:
                        real_val = numeric_val.real
                        if real_val == int(real_val):
                            formatted_solutions.append(str(int(real_val)))
                        else:
                            formatted_solutions.append(f"{real_val:.6g}")
                    else:
                        formatted_solutions.append(str(sol))
                else:
                    formatted_solutions.append(str(sol))

            return f"{var_name} = {', '.join(formatted_solutions)}"

    except SympifyError as e:
        return f"Error parsing equation: {str(e)}"
    except Exception as e:
        return f"Error solving equation: {str(e)}"


def _get_help() -> str:
    """Return help text."""
    return """Equation Solver Help:

BASIC USAGE:
  x + 2 = 5           -> x = 3
  2x + 3 = 11         -> x = 4
  x^2 - 4 = 0         -> x = -2, 2
  x^2 + 2x + 1 = 0    -> x = -1

WITH 'SOLVE' PREFIX:
  solve x + 2 = 5
  solve 3x - 9 = 0

IMPLICIT ZERO:
  x^2 - 9             -> solves x^2 - 9 = 0

MULTIPLE VARIABLES:
  solve for x: 2x + y = 10    -> x = 5 - y/2

SUPPORTED OPERATIONS:
  + - * / ^           (power: x^2 or x**2)
  sqrt(), sin(), cos(), tan(), log(), exp()

EXAMPLES:
  x/2 + 3 = 7         -> x = 8
  sqrt(x) = 4         -> x = 16
  x^2 + 5x + 6 = 0    -> x = -3, -2
  2^x = 8             -> x = 3"""


# Create the LangChain Tool wrapper
equation_solver_tool = Tool(
    name="equation_solver",
    func=solve_equation,
    description=(
        "Solve mathematical equations for unknown variables. "
        "\n\nEXAMPLES:"
        "\n- 'x + 2 = 5' -> x = 3"
        "\n- '2x + 3 = 11' -> x = 4"
        "\n- 'x^2 - 4 = 0' -> x = -2, 2"
        "\n- 'x^2 + 2x + 1 = 0' -> x = -1"
        "\n\nSupports linear, quadratic, and more complex equations. "
        "Use ^ or ** for powers. Implicit multiplication works (2x = 2*x)."
    )
)
