"""Equation solver — SymPy-powered equations, systems, matrices, and symbolic algebra."""

import re
import json
from typing import List, Optional, Type, Literal
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from sympy import (
        symbols, Eq, solve, sympify, SympifyError, Matrix,
        simplify as sym_simplify, expand as sym_expand, factor as sym_factor,
        diff as sym_diff, integrate as sym_integrate,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_KNOWN_FUNCTIONS = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs',
                     'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan'}


def _preprocess_equation(equation_str: str) -> str:
    """Convert ^ to ** and insert implicit multiplication for SymPy."""
    equation_str = equation_str.replace("^", "**")
    # 2x → 2*x
    equation_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation_str)
    # )x → )*x
    equation_str = re.sub(r'\)([a-zA-Z])', r')*\1', equation_str)
    # x( → x*( BUT skip known functions like sin(, cos(, etc.
    equation_str = re.sub(
        r'([a-zA-Z])\(',
        lambda m: m.group(0) if _is_function_prefix(equation_str, m.start()) else m.group(1) + '*(',
        equation_str,
    )
    return equation_str


def _is_function_prefix(s: str, pos: int) -> bool:
    """Check if position is the trailing char of a known function name."""
    for func in _KNOWN_FUNCTIONS:
        start = pos - len(func) + 1
        if start >= 0 and s[start:pos + 1] == func:
            # Make sure it's not part of a longer word
            if start == 0 or not s[start - 1].isalpha():
                return True
    return False


def _extract_variables(expression_str: str) -> List[str]:
    """Extract single-letter variable names, excluding reserved math names."""
    reserved = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs', 'pi', 'e'}
    potential_vars = re.findall(r'\b([a-zA-Z])\b', expression_str)
    variables = []
    seen = set()
    for var in potential_vars:
        if var.lower() not in reserved and var not in seen:
            variables.append(var)
            seen.add(var)
    return variables


def _parse_sympy_expr(expr_str: str, local_dict: dict):
    """Parse a string into a SymPy expression with implicit multiplication."""
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(expr_str, local_dict=local_dict, transformations=transformations)


def _format_number(val) -> str:
    """Format a SymPy value as a clean numeric string."""
    if val.is_number:
        numeric = complex(val.evalf())
        if numeric.imag == 0:
            real = numeric.real
            if real == int(real):
                return str(int(real))
            return f"{real:.6g}"
        return str(val)
    return str(val)


# ---------------------------------------------------------------------------
# Single equation (existing)
# ---------------------------------------------------------------------------

def _solve_single_equation(input_str: str, target_var: Optional[str] = None) -> str:
    """Solve a single equation for one variable (defaults to x)."""
    if "=" not in input_str:
        input_str = input_str + " = 0"

    parts = input_str.split("=")
    if len(parts) != 2:
        return "Error: Equation must have exactly one '=' sign"

    left_side = _preprocess_equation(parts[0].strip())
    right_side = _preprocess_equation(parts[1].strip())

    all_vars = _extract_variables(left_side + " " + right_side)
    if not all_vars:
        return "Error: No variables found in equation"

    sym_vars = {var: symbols(var) for var in all_vars}
    left_expr = _parse_sympy_expr(left_side, sym_vars)
    right_expr = _parse_sympy_expr(right_side, sym_vars)
    equation = Eq(left_expr, right_expr)

    if target_var and target_var in sym_vars:
        solve_var = sym_vars[target_var]
    elif len(all_vars) == 1:
        solve_var = sym_vars[all_vars[0]]
    elif 'x' in all_vars:
        solve_var = sym_vars['x']
    else:
        solve_var = sym_vars[all_vars[0]]

    solutions = solve(equation, solve_var)
    if not solutions:
        return f"No solution found for {solve_var}"

    var_name = str(solve_var)
    if len(solutions) == 1:
        return f"{var_name} = {_format_number(solutions[0])}"

    formatted = [_format_number(sol) for sol in solutions]
    return f"{var_name} = {', '.join(formatted)}"


# ---------------------------------------------------------------------------
# Systems of equations (new)
# ---------------------------------------------------------------------------

def _solve_system(input_str: str) -> str:
    """Solve a comma-separated system of equations for all variables."""
    equations_str = [eq.strip() for eq in input_str.split(",")]
    if len(equations_str) < 2:
        return "Error: System needs at least 2 equations separated by commas."

    all_vars_set = set()
    equations = []

    for eq_str in equations_str:
        if "=" not in eq_str:
            return f"Error: Each equation needs '='. Problem: '{eq_str}'"

        parts = eq_str.split("=")
        if len(parts) != 2:
            return f"Error: Invalid equation: '{eq_str}'"

        left = _preprocess_equation(parts[0].strip())
        right = _preprocess_equation(parts[1].strip())
        all_vars_set.update(_extract_variables(left + " " + right))
        equations.append((left, right))

    if not all_vars_set:
        return "Error: No variables found in system"

    sym_vars = {var: symbols(var) for var in sorted(all_vars_set)}

    sym_equations = []
    for left, right in equations:
        left_expr = _parse_sympy_expr(left, sym_vars)
        right_expr = _parse_sympy_expr(right, sym_vars)
        sym_equations.append(Eq(left_expr, right_expr))

    solve_for = [sym_vars[v] for v in sorted(all_vars_set)]
    solutions = solve(sym_equations, solve_for)

    if not solutions:
        return "No solution found for this system."

    # Solutions can be a dict (unique solution) or list of tuples (multiple)
    if isinstance(solutions, dict):
        lines = ["Solution:"]
        for var, val in solutions.items():
            lines.append(f"  {var} = {_format_number(val)}")
        return "\n".join(lines)
    elif isinstance(solutions, list):
        if len(solutions) == 1 and isinstance(solutions[0], tuple):
            lines = ["Solution:"]
            for var, val in zip(solve_for, solutions[0]):
                lines.append(f"  {var} = {_format_number(val)}")
            return "\n".join(lines)
        lines = [f"Found {len(solutions)} solutions:"]
        for i, sol in enumerate(solutions, 1):
            if isinstance(sol, tuple):
                parts = [f"{v} = {_format_number(s)}" for v, s in zip(solve_for, sol)]
                lines.append(f"  {i}. {', '.join(parts)}")
            else:
                lines.append(f"  {i}. {sol}")
        return "\n".join(lines)

    return f"Solution: {solutions}"


# ---------------------------------------------------------------------------
# Matrix operations (new)
# ---------------------------------------------------------------------------

def _parse_matrix(matrix_str: str) -> "Matrix":
    """Parse a string like '[[1,2],[3,4]]' into a SymPy Matrix."""
    matrix_str = matrix_str.strip()
    try:
        data = json.loads(matrix_str)
        return Matrix(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try SymPy parsing
    try:
        cleaned = matrix_str.replace(" ", "")
        data = eval(cleaned, {"__builtins__": {}})
        return Matrix(data)
    except Exception:
        raise ValueError(f"Cannot parse matrix: '{matrix_str}'. Use format: [[1,2],[3,4]]")


def _format_matrix(m: "Matrix") -> str:
    """Format a SymPy Matrix as a readable nested-bracket string."""
    rows = m.tolist()
    # Format each element
    formatted_rows = []
    for row in rows:
        formatted_row = [_format_number(sympify(val)) if not isinstance(val, (int, float)) else
                         str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
                         for val in row]
        formatted_rows.append("[" + ", ".join(formatted_row) + "]")
    return "[" + ", ".join(formatted_rows) + "]"


def _handle_matrix_operation(op: str, input_str: str) -> str:
    """Dispatch matrix operations (multiply, inverse, det, eigenvalues, etc.)."""
    try:
        if op == "multiply":
            # Parse two matrices separated by *
            parts = input_str.split("*")
            if len(parts) != 2:
                return "Error: Matrix multiplication needs two matrices separated by *. Example: [[1,2],[3,4]] * [[5,6],[7,8]]"
            m1 = _parse_matrix(parts[0])
            m2 = _parse_matrix(parts[1])
            result = m1 * m2
            return f"Result:\n{_format_matrix(result)}"

        m = _parse_matrix(input_str)

        if op == "inverse":
            if m.det() == 0:
                return "Error: Matrix is singular (determinant = 0), cannot invert."
            result = m.inv()
            return f"Inverse:\n{_format_matrix(result)}"

        elif op == "determinant":
            det = m.det()
            return f"Determinant = {_format_number(det)}"

        elif op == "eigenvalues":
            eigenvals = m.eigenvals()
            lines = ["Eigenvalues:"]
            for val, mult in eigenvals.items():
                val_str = _format_number(val)
                if mult > 1:
                    lines.append(f"  {val_str} (multiplicity {mult})")
                else:
                    lines.append(f"  {val_str}")
            return "\n".join(lines)

        elif op == "transpose":
            result = m.T
            return f"Transpose:\n{_format_matrix(result)}"

        elif op == "rank":
            return f"Rank = {m.rank()}"

        elif op == "rref":
            rref_matrix, pivots = m.rref()
            return f"Row echelon form:\n{_format_matrix(rref_matrix)}\nPivot columns: {list(pivots)}"

        else:
            return f"Error: Unknown matrix operation '{op}'"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error in matrix operation: {str(e)}"


# ---------------------------------------------------------------------------
# Symbolic algebra (new)
# ---------------------------------------------------------------------------

def _handle_symbolic(op: str, input_str: str) -> str:
    """Handle symbolic algebra (simplify, expand, factor, derivative, integrate)."""
    try:
        preprocessed = _preprocess_equation(input_str.strip())
        all_vars = _extract_variables(preprocessed)
        sym_vars = {var: symbols(var) for var in all_vars}

        expr = _parse_sympy_expr(preprocessed, sym_vars)

        # Determine the primary variable (for calculus operations)
        if 'x' in sym_vars:
            primary_var = sym_vars['x']
        elif all_vars:
            primary_var = sym_vars[all_vars[0]]
        else:
            primary_var = None

        if op == "simplify":
            result = sym_simplify(expr)
            return f"Simplified: {result}"

        elif op == "expand":
            result = sym_expand(expr)
            return f"Expanded: {result}"

        elif op == "factor":
            result = sym_factor(expr)
            return f"Factored: {result}"

        elif op in ("derivative", "diff"):
            if primary_var is None:
                return "Error: No variable found for differentiation"
            result = sym_diff(expr, primary_var)
            return f"d/d{primary_var}({expr}) = {result}"

        elif op in ("integrate", "integral"):
            if primary_var is None:
                return "Error: No variable found for integration"
            result = sym_integrate(expr, primary_var)
            return f"∫({expr}) d{primary_var} = {result} + C"

        else:
            return f"Error: Unknown operation '{op}'"

    except SympifyError as e:
        return f"Error parsing expression: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _get_help() -> str:
    """Return help text."""
    return """Equation Solver & Symbolic Math Help:

EQUATIONS:
  x + 2 = 5              -> x = 3
  x^2 - 4 = 0            -> x = -2, 2
  solve for x: 2x + y = 10

SYSTEMS OF EQUATIONS:
  system: x + y = 5, 2x - y = 1
  system: a + b + c = 6, 2a - b = 1, b + c = 4

MATRIX OPERATIONS:
  matrix: [[1,2],[3,4]] * [[5,6],[7,8]]    (multiply)
  inverse: [[1,2],[3,4]]
  determinant: [[1,2],[3,4]]
  eigenvalues: [[4,-2],[1,1]]
  transpose: [[1,2,3],[4,5,6]]
  rank: [[1,2],[2,4]]
  rref: [[1,2,3],[4,5,6]]

SYMBOLIC ALGEBRA:
  simplify: (x^2 - 1)/(x - 1)
  expand: (x + 1)^3
  factor: x^2 + 5x + 6

CALCULUS:
  derivative: x^3 + 2x
  integrate: x^2 + 3x"""


# ---------------------------------------------------------------------------
# BaseTool subclass
# ---------------------------------------------------------------------------

class EquationSolverInput(BaseModel):
    operation: Literal[
        "solve", "system", "simplify", "expand", "factor",
        "derivative", "integral",
        "matrix_det", "matrix_mul", "matrix_inv",
        "matrix_transpose", "matrix_rank", "matrix_rref",
        "eigenvalues"
    ] = Field(default="solve", description="Math operation to perform")
    expression: str = Field(description="The equation, matrix, or expression to operate on")


class EquationSolverTool(BaseTool):
    name: str = "equation_solver"
    description: str = (
        "Solve equations, systems of equations, matrix operations, and symbolic algebra. "
        "\n\nEQUATIONS: 'x^2 - 4 = 0', 'solve for x: 2x + y = 10'"
        "\n\nSYSTEMS: 'system: x + y = 5, 2x - y = 1'"
        "\n\nMATRIX: 'inverse: [[1,2],[3,4]]', 'determinant: [[1,2],[3,4]]', 'eigenvalues: [[4,-2],[1,1]]'"
        "\n\nALGEBRA: 'simplify: expr', 'expand: expr', 'factor: expr'"
        "\n\nCALCULUS: 'derivative: x^3 + 2x', 'integrate: x^2'"
    )
    args_schema: Type[BaseModel] = EquationSolverInput

    def _run(self, operation: str = "solve", expression: str = "") -> str:
        if not SYMPY_AVAILABLE:
            return "Error: SymPy library is not installed. Run: pip install sympy"

        expression = expression.strip()
        if not expression:
            return "Error: No expression provided."
        if expression.lower() in ("help", "?"):
            return _get_help()

        op = operation.lower()
        if op == "system":
            return _solve_system(expression)
        elif op in ("matrix_det",):
            return _handle_matrix_operation("determinant", expression)
        elif op in ("matrix_mul",):
            return _handle_matrix_operation("multiply", expression)
        elif op in ("matrix_inv",):
            return _handle_matrix_operation("inverse", expression)
        elif op == "matrix_transpose":
            return _handle_matrix_operation("transpose", expression)
        elif op == "matrix_rank":
            return _handle_matrix_operation("rank", expression)
        elif op == "matrix_rref":
            return _handle_matrix_operation("rref", expression)
        elif op == "eigenvalues":
            return _handle_matrix_operation("eigenvalues", expression)
        elif op in ("simplify", "expand", "factor", "derivative", "integral"):
            return _handle_symbolic(op, expression)
        else:
            # Default: solve as equation
            try:
                return _solve_single_equation(expression)
            except SympifyError as e:
                return f"Error parsing equation: {str(e)}"
            except Exception as e:
                return f"Error solving equation: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


equation_solver_tool = EquationSolverTool()

def solve_equation(input_str: str) -> str:
    """Parse a string with optional prefix and solve. Used by tests and CLI."""
    if not input_str or not input_str.strip():
        return "Error: Empty input"
    text = input_str.strip()
    low = text.lower()
    if low in ("help", "?"):
        return equation_solver_tool._run(operation="solve", expression="help")
    # Detect operation from prefix
    for prefix, op in [
        ("system:", "system"), ("simplify:", "simplify"), ("expand:", "expand"),
        ("factor:", "factor"), ("derivative:", "derivative"), ("diff:", "derivative"),
        ("integrate:", "integral"), ("integral:", "integral"),
        ("matrix:", "matrix_mul"), ("multiply:", "matrix_mul"),
        ("inverse:", "matrix_inv"), ("inv:", "matrix_inv"),
        ("determinant:", "matrix_det"), ("det:", "matrix_det"),
        ("eigenvalues:", "eigenvalues"), ("eigen:", "eigenvalues"),
        ("transpose:", "matrix_transpose"), ("rank:", "matrix_rank"),
        ("rref:", "matrix_rref"),
    ]:
        if low.startswith(prefix):
            return equation_solver_tool._run(operation=op, expression=text[len(prefix):].strip())
    return equation_solver_tool._run(operation="solve", expression=text)
