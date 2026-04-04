"""Step-by-step math solver for student-facing calculations.

Generates pedagogical breakdowns for:
- Complex arithmetic (order of operations)
- Derivatives (power rule, chain rule, trig)
- Integrals (term-by-term, definite with bounds)
- Equation solving (linear rearrangement, quadratic formula)
- Matrix operations (determinant, multiply, inverse, transpose, addition)

Uses SymPy for symbolic math. Returns formatted multi-line strings
that the LLM agent streams to students.
"""

import ast
import json
import re
from typing import List, Optional, Tuple

try:
    from sympy import (
        symbols, Symbol, Eq, solve, sympify, Matrix,
        simplify as sym_simplify, expand as sym_expand,
        factor as sym_factor, diff as sym_diff,
        integrate as sym_integrate, Rational,
        Add, Mul, Pow, sin, cos, tan, exp, log, sqrt,
        oo, pi as sym_pi, E as sym_E,
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Reuse helpers from equation_solver_tool
from src.tools.equation_solver_tool import (
    _preprocess_equation, _extract_variables, _parse_sympy_expr,
    _parse_matrix, _format_matrix, _format_number,
)


# ============================================================================
# OPERATION DETECTION
# ============================================================================

def detect_operation(input_str: str) -> Tuple[str, str]:
    """Classify input into an operation type.

    Returns:
        (operation_type, cleaned_input) where operation_type is one of:
        "simple", "complex_arithmetic", "derivative", "integral",
        "solve", "matrix_det", "matrix_mul", "matrix_inv",
        "matrix_trans", "matrix_add", "passthrough"
    """
    s = input_str.strip()
    lower = s.lower()

    # Commands that the calculator handles directly
    if lower in ("variables", "vars", "list", "clear", "clear variables",
                 "clear vars", "help", "?"):
        return ("passthrough", s)

    # Variable assignment
    if re.match(r'(?:set\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+', s):
        return ("passthrough", s)

    # --- Matrix operations ---
    has_matrix = "[[" in s

    if has_matrix:
        # Matrix multiply: [[...]] * [[...]]
        if re.search(r'\]\]\s*\*\s*\[\[', s):
            return ("matrix_mul", s)

        # Matrix addition: [[...]] + [[...]]
        if re.search(r'\]\]\s*\+\s*\[\[', s):
            return ("matrix_add", s)

        # Keyword-based matrix ops
        if re.match(r'(?:determinant|det)\b', lower):
            expr = re.sub(r'^(?:determinant|det)\s*', '', s, flags=re.IGNORECASE).strip()
            return ("matrix_det", expr)

        if re.match(r'(?:inverse|inv)\b', lower):
            expr = re.sub(r'^(?:inverse|inv)\s*', '', s, flags=re.IGNORECASE).strip()
            return ("matrix_inv", expr)

        if re.match(r'transpose\b', lower):
            expr = re.sub(r'^transpose\s*', '', s, flags=re.IGNORECASE).strip()
            return ("matrix_trans", expr)

    # --- Calculus ---
    if re.match(r'(?:derivative|diff|d/d[a-z])\b', lower):
        expr = re.sub(r'^(?:derivative|diff|d/d[a-z])\s*(?:of\s+)?', '', s, flags=re.IGNORECASE).strip()
        return ("derivative", expr)

    if re.match(r'(?:integrate|integral)\b', lower):
        expr = re.sub(r'^(?:integrate|integral)\s*(?:of\s+)?', '', s, flags=re.IGNORECASE).strip()
        return ("integral", expr)

    # --- Equation solving ---
    if lower.startswith("solve"):
        expr = s[5:].strip()
        return ("solve", expr)

    # Expression with = and a variable (likely an equation)
    if "=" in s and re.search(r'[a-zA-Z]', s):
        # But not variable assignment (already caught above)
        return ("solve", s)

    # --- Complex arithmetic ---
    if _is_complex_arithmetic(s):
        return ("complex_arithmetic", s)

    return ("simple", s)


def _is_complex_arithmetic(expr: str) -> bool:
    """Check if an expression is complex enough to warrant step-by-step."""
    # Count binary operators (excluding unary minus at start or after open paren)
    cleaned = re.sub(r'(?:^|(?<=\())-', '', expr)  # remove unary minus
    ops = re.findall(r'[\+\-\*/\%]|\*\*', cleaned)
    if len(ops) >= 3:
        return True

    # Check parenthesis nesting depth
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ')':
            depth -= 1
    if max_depth >= 2:
        return True

    return False


# ============================================================================
# STEP-BY-STEP SOLVER
# ============================================================================

class StepByStepSolver:
    """Generates step-by-step math solutions for students."""

    def solve(self, operation: str, expr: str) -> str:
        """Main dispatcher — returns a formatted step-by-step solution string."""
        if not SYMPY_AVAILABLE and operation not in ("complex_arithmetic",):
            return (f"Error: SymPy is not installed (needed for {operation}). "
                    "Run: pip install sympy")

        try:
            if operation == "complex_arithmetic":
                return self._arithmetic_steps(expr)
            elif operation == "derivative":
                return self._derivative_steps(expr)
            elif operation == "integral":
                return self._integral_steps(expr)
            elif operation == "solve":
                return self._solve_equation_steps(expr)
            elif operation == "matrix_det":
                return self._matrix_determinant_steps(expr)
            elif operation == "matrix_mul":
                return self._matrix_multiply_steps(expr)
            elif operation == "matrix_inv":
                return self._matrix_inverse_steps(expr)
            elif operation == "matrix_trans":
                return self._matrix_transpose_steps(expr)
            elif operation == "matrix_add":
                return self._matrix_add_steps(expr)
            else:
                return f"Error: Unknown operation '{operation}'"
        except Exception as e:
            return f"Error: {str(e)}"

    # ------------------------------------------------------------------
    # Complex arithmetic
    # ------------------------------------------------------------------

    def _arithmetic_steps(self, expr: str) -> str:
        """Show order-of-operations steps for complex arithmetic."""
        lines = [f"Step-by-step solution for: {expr}", ""]

        try:
            tree = ast.parse(expr, mode='eval')
        except SyntaxError:
            return f"Error: Invalid syntax in expression"

        steps = []
        current = expr
        result = self._eval_ast_steps(tree.body, steps)

        step_num = 1
        for description, intermediate in steps:
            lines.append(f"Step {step_num}: {description}")
            lines.append(f"  = {intermediate}")
            lines.append("")
            step_num += 1

        final = self._format_result(result)
        lines.append(f"Result: {final}")
        return "\n".join(lines)

    def _eval_ast_steps(self, node, steps: list) -> float:
        """Recursively evaluate an AST node, recording steps."""
        if isinstance(node, ast.Constant):
            return float(node.value)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_steps(node.operand, steps)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand

        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_steps(node.left, steps)
            right = self._eval_ast_steps(node.right, steps)
            op_symbol = self._ast_op_symbol(node.op)

            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                result = left / right
            elif isinstance(node.op, ast.Pow):
                result = left ** right
            elif isinstance(node.op, ast.Mod):
                result = left % right
            elif isinstance(node.op, ast.FloorDiv):
                result = left // right
            else:
                raise ValueError(f"Unsupported operator")

            left_str = self._format_result(left)
            right_str = self._format_result(right)
            result_str = self._format_result(result)

            # Only record step if both sides are evaluated (not trivial)
            if not (isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant)
                    and len(steps) == 0):
                steps.append((
                    f"Calculate {left_str} {op_symbol} {right_str}",
                    result_str
                ))
            elif isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                steps.append((
                    f"Calculate {left_str} {op_symbol} {right_str}",
                    result_str
                ))

            return result

        elif isinstance(node, ast.Call):
            # Function calls like sqrt(16)
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = [self._eval_ast_steps(arg, steps) for arg in node.args]
                import math
                safe_funcs = {
                    "sqrt": math.sqrt, "abs": abs, "round": round,
                    "sin": math.sin, "cos": math.cos, "tan": math.tan,
                    "log": math.log, "log10": math.log10, "log2": math.log2,
                    "exp": math.exp, "factorial": math.factorial,
                    "ceil": math.ceil, "floor": math.floor,
                    "min": min, "max": max,
                }
                if func_name in safe_funcs:
                    result = safe_funcs[func_name](*args)
                    args_str = ", ".join(self._format_result(a) for a in args)
                    steps.append((
                        f"Evaluate {func_name}({args_str})",
                        self._format_result(result)
                    ))
                    return float(result)

            raise ValueError(f"Unknown function in expression")

        elif isinstance(node, ast.Name):
            # Handle constants
            import math
            constants = {"pi": math.pi, "e": math.e, "tau": math.tau}
            if node.id in constants:
                return constants[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        raise ValueError(f"Cannot evaluate expression")

    def _ast_op_symbol(self, op) -> str:
        """Get the symbol for an AST operator."""
        symbols_map = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*",
            ast.Div: "/", ast.Pow: "**", ast.Mod: "%",
            ast.FloorDiv: "//",
        }
        return symbols_map.get(type(op), "?")

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def _derivative_steps(self, expr_str: str) -> str:
        """Show step-by-step differentiation."""
        preprocessed = _preprocess_equation(expr_str)
        all_vars = _extract_variables(preprocessed)
        sym_vars = {var: symbols(var) for var in all_vars}

        if not all_vars:
            return "Error: No variable found for differentiation"

        # Pick primary variable
        if 'x' in sym_vars:
            var = sym_vars['x']
        else:
            var = sym_vars[all_vars[0]]

        expr = _parse_sympy_expr(preprocessed, sym_vars)
        var_name = str(var)

        lines = [f"Step-by-step solution for: d/d{var_name}({expr})", ""]

        # Split into additive terms
        terms = Add.make_args(expr)
        step_num = 1

        if len(terms) > 1:
            lines.append(f"Step {step_num}: Break into terms (sum rule)")
            for t in terms:
                lines.append(f"  d/d{var_name}({t})")
            lines.append("")
            step_num += 1

        term_derivatives = []
        for term in terms:
            deriv = sym_diff(term, var)
            rule = self._identify_rule(term, var)
            lines.append(f"Step {step_num}: Differentiate {term}  [{rule}]")
            lines.append(f"  = {deriv}")
            lines.append("")
            term_derivatives.append(deriv)
            step_num += 1

        # Combine
        final = sym_diff(expr, var)
        simplified = sym_simplify(final)

        if len(terms) > 1:
            lines.append(f"Step {step_num}: Combine all terms")
            lines.append(f"  = {final}")
            lines.append("")
            step_num += 1

        if simplified != final:
            lines.append(f"Step {step_num}: Simplify")
            lines.append(f"  = {simplified}")
            lines.append("")

        lines.append(f"Result: d/d{var_name}({expr}) = {simplified}")
        return "\n".join(lines)

    def _identify_rule(self, term, var) -> str:
        """Identify which differentiation rule applies to a term."""
        var_sym = var if isinstance(var, Symbol) else symbols(str(var))

        if not term.has(var_sym):
            return "constant rule: derivative of a constant is 0"

        # Check for power: x**n or c*x**n
        if term.is_Pow or (term.is_Mul and any(f.is_Pow for f in term.args)):
            return "power rule: d/dx(x^n) = n*x^(n-1)"

        # Simple x term
        if term == var_sym or (term.is_Mul and var_sym in term.args):
            return "power rule: d/dx(x) = 1"

        # Trig functions
        func_rules = {
            sin: "trig rule: d/dx(sin(x)) = cos(x)",
            cos: "trig rule: d/dx(cos(x)) = -sin(x)",
            tan: "trig rule: d/dx(tan(x)) = sec^2(x)",
            exp: "exponential rule: d/dx(e^x) = e^x",
            log: "logarithm rule: d/dx(ln(x)) = 1/x",
        }
        for func, rule in func_rules.items():
            if term.has(func):
                # Check if argument is not just x (chain rule)
                for arg in term.atoms(func):
                    inner = arg.args[0]
                    if inner != var_sym and inner.has(var_sym):
                        return rule + " + chain rule"
                return rule

        return "differentiation"

    # ------------------------------------------------------------------
    # Integrals
    # ------------------------------------------------------------------

    def _integral_steps(self, expr_str: str) -> str:
        """Show step-by-step integration."""
        # Parse bounds if present: "x^2 from 0 to 5" or "x^2 dx from 0 to 5"
        bounds = None
        bounds_match = re.search(r'\bfrom\s+([\d.\-]+)\s+to\s+([\d.\-]+)', expr_str)
        if bounds_match:
            lower_bound = float(bounds_match.group(1))
            upper_bound = float(bounds_match.group(2))
            bounds = (lower_bound, upper_bound)
            expr_str = expr_str[:bounds_match.start()].strip()

        # Remove trailing "dx" etc.
        expr_str = re.sub(r'\s*d[a-z]\s*$', '', expr_str).strip()

        preprocessed = _preprocess_equation(expr_str)
        all_vars = _extract_variables(preprocessed)
        sym_vars = {var: symbols(var) for var in all_vars}

        if not all_vars:
            return "Error: No variable found for integration"

        if 'x' in sym_vars:
            var = sym_vars['x']
        else:
            var = sym_vars[all_vars[0]]

        expr = _parse_sympy_expr(preprocessed, sym_vars)
        var_name = str(var)

        if bounds:
            title = f"Step-by-step solution for: integral of {expr} d{var_name} from {self._format_result(bounds[0])} to {self._format_result(bounds[1])}"
        else:
            title = f"Step-by-step solution for: integral of {expr} d{var_name}"

        lines = [title, ""]

        # Split into additive terms
        terms = Add.make_args(expr)
        step_num = 1

        if len(terms) > 1:
            lines.append(f"Step {step_num}: Break into terms (sum rule)")
            for t in terms:
                lines.append(f"  integral of {t} d{var_name}")
            lines.append("")
            step_num += 1

        term_integrals = []
        for term in terms:
            antideriv = sym_integrate(term, var)
            rule = self._identify_integral_rule(term, var)
            lines.append(f"Step {step_num}: Integrate {term}  [{rule}]")
            lines.append(f"  = {antideriv}")
            lines.append("")
            term_integrals.append(antideriv)
            step_num += 1

        antiderivative = sym_integrate(expr, var)

        if len(terms) > 1:
            lines.append(f"Step {step_num}: Combine all terms")
            lines.append(f"  F({var_name}) = {antiderivative}")
            lines.append("")
            step_num += 1

        if bounds:
            a, b = bounds
            sym_a, sym_b = sympify(a), sympify(b)
            f_b = antiderivative.subs(var, sym_b)
            f_a = antiderivative.subs(var, sym_a)

            lines.append(f"Step {step_num}: Evaluate F({self._format_result(b)}) - F({self._format_result(a)})")
            lines.append(f"  F({self._format_result(b)}) = {_format_number(sympify(f_b))}")
            lines.append(f"  F({self._format_result(a)}) = {_format_number(sympify(f_a))}")
            lines.append(f"  = {_format_number(sympify(f_b))} - ({_format_number(sympify(f_a))})")
            lines.append("")
            step_num += 1

            result = f_b - f_a
            simplified = sym_simplify(result)
            lines.append(f"Result: {_format_number(sympify(simplified))}")
        else:
            lines.append(f"Result: {antiderivative} + C")

        return "\n".join(lines)

    def _identify_integral_rule(self, term, var) -> str:
        """Identify which integration rule applies."""
        var_sym = var if isinstance(var, Symbol) else symbols(str(var))

        if not term.has(var_sym):
            return "constant rule: integral of c = c*x"

        if term == var_sym:
            return "power rule: integral of x = x^2/2"

        if term.is_Pow and term.args[0] == var_sym:
            return "power rule: integral of x^n = x^(n+1)/(n+1)"

        if term.is_Mul:
            # c * x^n
            has_power = any(f.is_Pow and f.args[0] == var_sym for f in term.args)
            if has_power or var_sym in term.args:
                return "power rule (with coefficient)"

        func_rules = {
            sin: "trig rule: integral of sin(x) = -cos(x)",
            cos: "trig rule: integral of cos(x) = sin(x)",
            exp: "exponential rule: integral of e^x = e^x",
        }
        for func, rule in func_rules.items():
            if term.has(func):
                return rule

        return "integration"

    # ------------------------------------------------------------------
    # Equation solving
    # ------------------------------------------------------------------

    def _solve_equation_steps(self, expr_str: str) -> str:
        """Show step-by-step equation solving."""
        if "=" not in expr_str:
            expr_str = expr_str + " = 0"

        parts = expr_str.split("=")
        if len(parts) != 2:
            return "Error: Equation must have exactly one '=' sign"

        left_str = parts[0].strip()
        right_str = parts[1].strip()

        left_pre = _preprocess_equation(left_str)
        right_pre = _preprocess_equation(right_str)

        all_vars = _extract_variables(left_pre + " " + right_pre)
        if not all_vars:
            return "Error: No variables found in equation"

        sym_vars = {v: symbols(v) for v in all_vars}

        if 'x' in sym_vars:
            var = sym_vars['x']
        else:
            var = sym_vars[all_vars[0]]

        var_name = str(var)

        left_expr = _parse_sympy_expr(left_pre, sym_vars)
        right_expr = _parse_sympy_expr(right_pre, sym_vars)

        lines = [f"Step-by-step solution for: {left_str} = {right_str}", ""]
        step_num = 1

        # Move everything to one side
        full_expr = left_expr - right_expr
        full_expanded = sym_expand(full_expr)

        lines.append(f"Step {step_num}: Rearrange to standard form (= 0)")
        lines.append(f"  {full_expanded} = 0")
        lines.append("")
        step_num += 1

        # Determine degree
        poly_expr = sym_expand(full_expr)
        degree = poly_expr.as_poly(var).degree() if poly_expr.is_polynomial(var) else None

        if degree == 1:
            # Linear equation: show isolation steps
            coeffs = poly_expr.as_poly(var).all_coeffs()
            a_coeff, b_coeff = coeffs[0], coeffs[1] if len(coeffs) > 1 else 0

            if b_coeff != 0:
                lines.append(f"Step {step_num}: Isolate the variable term")
                lines.append(f"  {a_coeff}*{var_name} = {-b_coeff}")
                lines.append("")
                step_num += 1

            if a_coeff != 1:
                lines.append(f"Step {step_num}: Divide both sides by {a_coeff}")
                solution = -b_coeff / a_coeff
                lines.append(f"  {var_name} = {-b_coeff}/{a_coeff} = {_format_number(sympify(solution))}")
                lines.append("")
                step_num += 1

            solutions = solve(Eq(left_expr, right_expr), var)
            lines.append(f"Result: {var_name} = {_format_number(solutions[0])}")

        elif degree == 2:
            # Quadratic equation
            coeffs = poly_expr.as_poly(var).all_coeffs()
            a, b, c = (coeffs + [0, 0, 0])[:3]

            lines.append(f"Step {step_num}: Identify coefficients (a{var_name}^2 + b{var_name} + c = 0)")
            lines.append(f"  a = {a}, b = {b}, c = {c}")
            lines.append("")
            step_num += 1

            discriminant = b**2 - 4*a*c
            lines.append(f"Step {step_num}: Calculate discriminant (b^2 - 4ac)")
            lines.append(f"  D = ({b})^2 - 4*({a})*({c}) = {discriminant}")
            lines.append("")
            step_num += 1

            if discriminant > 0:
                lines.append(f"Step {step_num}: D > 0, so two real solutions")
                lines.append(f"  {var_name} = (-b +/- sqrt(D)) / (2a)")
                lines.append("")
                step_num += 1
            elif discriminant == 0:
                lines.append(f"Step {step_num}: D = 0, so one repeated solution")
                lines.append(f"  {var_name} = -b / (2a)")
                lines.append("")
                step_num += 1
            else:
                lines.append(f"Step {step_num}: D < 0, so two complex solutions")
                lines.append("")
                step_num += 1

            # Try factoring
            factored = sym_factor(poly_expr)
            if factored != poly_expr:
                lines.append(f"Step {step_num}: Factor the expression")
                lines.append(f"  {factored} = 0")
                lines.append("")
                step_num += 1

            solutions = solve(Eq(left_expr, right_expr), var)
            formatted = [_format_number(s) for s in solutions]
            lines.append(f"Result: {var_name} = {', '.join(formatted)}")

        else:
            # General case — just solve and show result
            factored = sym_factor(poly_expr)
            if factored != poly_expr:
                lines.append(f"Step {step_num}: Factor the expression")
                lines.append(f"  {factored} = 0")
                lines.append("")
                step_num += 1

            solutions = solve(Eq(left_expr, right_expr), var)
            if solutions:
                formatted = [_format_number(s) for s in solutions]
                lines.append(f"Result: {var_name} = {', '.join(formatted)}")
            else:
                lines.append("Result: No solution found")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Matrix operations
    # ------------------------------------------------------------------

    def _matrix_determinant_steps(self, matrix_str: str) -> str:
        """Show step-by-step determinant calculation."""
        m = _parse_matrix(matrix_str)
        rows, cols = m.shape

        if rows != cols:
            return f"Error: Determinant requires a square matrix. Got {rows}x{cols}."

        data = m.tolist()
        lines = [f"Step-by-step solution for: determinant of {_format_matrix(m)}", ""]

        if rows == 1:
            lines.append(f"Step 1: 1x1 matrix — determinant is the single element")
            lines.append(f"Result: {data[0][0]}")

        elif rows == 2:
            a, b = data[0]
            c, d = data[1]
            lines.append(f"Step 1: Apply 2x2 determinant formula: ad - bc")
            lines.append(f"  a={a}, b={b}, c={c}, d={d}")
            lines.append("")
            lines.append(f"Step 2: Calculate")
            lines.append(f"  ({a})*({d}) - ({b})*({c})")
            lines.append(f"  = {a*d} - {b*c}")
            det = a*d - b*c
            lines.append(f"  = {det}")
            lines.append("")
            lines.append(f"Result: det = {det}")

        elif rows == 3:
            lines.append(f"Step 1: Use cofactor expansion along the first row")
            lines.append("")

            det = 0
            step_num = 2
            for j in range(3):
                sign = (-1) ** j
                sign_str = "+" if sign > 0 else "-"
                element = data[0][j]

                # Minor matrix (2x2)
                minor_rows = []
                for i in range(1, 3):
                    minor_row = [data[i][k] for k in range(3) if k != j]
                    minor_rows.append(minor_row)

                minor_det = minor_rows[0][0] * minor_rows[1][1] - minor_rows[0][1] * minor_rows[1][0]
                cofactor = sign * element * minor_det
                det += cofactor

                lines.append(f"Step {step_num}: Cofactor C(1,{j+1}): {sign_str} {abs(element)} * det({minor_rows})")
                lines.append(f"  Minor determinant = ({minor_rows[0][0]})*({minor_rows[1][1]}) - ({minor_rows[0][1]})*({minor_rows[1][0]}) = {minor_det}")
                lines.append(f"  Cofactor = {sign_str} {abs(element)} * {minor_det} = {cofactor}")
                lines.append("")
                step_num += 1

            lines.append(f"Step {step_num}: Sum all cofactors")
            cofactor_strs = []
            for j in range(3):
                sign = (-1) ** j
                element = data[0][j]
                minor_rows = []
                for i in range(1, 3):
                    minor_row = [data[i][k] for k in range(3) if k != j]
                    minor_rows.append(minor_row)
                minor_det = minor_rows[0][0] * minor_rows[1][1] - minor_rows[0][1] * minor_rows[1][0]
                cofactor_strs.append(str(sign * element * minor_det))
            lines.append(f"  = {' + '.join(cofactor_strs)}")
            lines.append("")
            lines.append(f"Result: det = {det}")

        else:
            # For 4x4+, show cofactor expansion concept but use SymPy for result
            lines.append(f"Step 1: Use cofactor expansion along the first row")
            lines.append(f"  (expanding a {rows}x{rows} matrix)")
            lines.append("")
            det = m.det()
            lines.append(f"Step 2: Computing cofactors recursively...")
            lines.append("")
            lines.append(f"Result: det = {_format_number(det)}")

        return "\n".join(lines)

    def _matrix_multiply_steps(self, expr_str: str) -> str:
        """Show step-by-step matrix multiplication."""
        parts = re.split(r'\]\]\s*\*\s*\[\[', expr_str)
        if len(parts) != 2:
            return "Error: Use format [[1,2],[3,4]] * [[5,6],[7,8]]"

        a_str = parts[0] + "]]"
        b_str = "[[" + parts[1]

        m1 = _parse_matrix(a_str)
        m2 = _parse_matrix(b_str)

        r1, c1 = m1.shape
        r2, c2 = m2.shape

        if c1 != r2:
            return f"Error: Cannot multiply {r1}x{c1} by {r2}x{c2}. Inner dimensions must match."

        a = m1.tolist()
        b = m2.tolist()

        lines = [f"Step-by-step solution for: {_format_matrix(m1)} * {_format_matrix(m2)}", ""]
        lines.append(f"Step 1: Verify dimensions: ({r1}x{c1}) * ({r2}x{c2}) = ({r1}x{c2})")
        lines.append("")

        result = [[0]*c2 for _ in range(r1)]
        step_num = 2

        # For small matrices, show each element calculation
        if r1 * c2 <= 16:  # Up to 4x4 result
            for i in range(r1):
                for j in range(c2):
                    products = []
                    product_strs = []
                    for k in range(c1):
                        val = a[i][k] * b[k][j]
                        products.append(val)
                        product_strs.append(f"({a[i][k]})*({b[k][j]})")
                    total = sum(products)
                    result[i][j] = total

                    lines.append(f"Step {step_num}: Element [{i+1},{j+1}] = Row {i+1} . Column {j+1}")
                    lines.append(f"  = {' + '.join(product_strs)}")
                    lines.append(f"  = {' + '.join(str(p) for p in products)} = {total}")
                    lines.append("")
                    step_num += 1
        else:
            lines.append(f"Step {step_num}: Computing dot products for each element...")
            lines.append(f"  ({r1}x{c2} = {r1*c2} elements)")
            lines.append("")

        result_matrix = m1 * m2
        lines.append(f"Result: {_format_matrix(result_matrix)}")
        return "\n".join(lines)

    def _matrix_inverse_steps(self, matrix_str: str) -> str:
        """Show step-by-step matrix inversion."""
        m = _parse_matrix(matrix_str)
        rows, cols = m.shape

        if rows != cols:
            return f"Error: Inverse requires a square matrix. Got {rows}x{cols}."

        data = m.tolist()
        lines = [f"Step-by-step solution for: inverse of {_format_matrix(m)}", ""]

        det = m.det()
        step_num = 1

        lines.append(f"Step {step_num}: Calculate the determinant")
        if rows == 2:
            a, b = data[0]
            c, d = data[1]
            lines.append(f"  det = ({a})*({d}) - ({b})*({c}) = {a*d} - {b*c} = {det}")
        else:
            lines.append(f"  det = {_format_number(det)}")
        lines.append("")
        step_num += 1

        if det == 0:
            lines.append(f"Step {step_num}: Determinant is 0 — matrix is singular (not invertible)")
            lines.append("")
            lines.append("Result: No inverse exists (singular matrix)")
            return "\n".join(lines)

        if rows == 2:
            a, b = data[0]
            c, d = data[1]
            lines.append(f"Step {step_num}: Apply 2x2 inverse formula: (1/det) * [[d, -b], [-c, a]]")
            lines.append(f"  = (1/{det}) * [[{d}, {-b}], [{-c}, {a}]]")
            lines.append("")
            step_num += 1

            inv = m.inv()
            lines.append(f"Step {step_num}: Multiply each element by 1/{det}")
            lines.append("")
            lines.append(f"Result: {_format_matrix(inv)}")
        else:
            lines.append(f"Step {step_num}: Compute adjugate (matrix of cofactors, transposed)")
            lines.append("")
            step_num += 1

            inv = m.inv()
            lines.append(f"Step {step_num}: Multiply adjugate by 1/det = 1/{_format_number(det)}")
            lines.append("")
            lines.append(f"Result: {_format_matrix(inv)}")

        return "\n".join(lines)

    def _matrix_transpose_steps(self, matrix_str: str) -> str:
        """Show step-by-step matrix transposition."""
        m = _parse_matrix(matrix_str)
        rows, cols = m.shape

        lines = [f"Step-by-step solution for: transpose of {_format_matrix(m)}", ""]
        lines.append(f"Step 1: Swap rows and columns ({rows}x{cols} becomes {cols}x{rows})")
        lines.append(f"  Row i becomes Column i")
        lines.append("")

        data = m.tolist()
        if rows * cols <= 16:
            lines.append(f"Step 2: Map each element")
            for i in range(rows):
                for j in range(cols):
                    lines.append(f"  Element [{i+1},{j+1}] = {data[i][j]} -> [{j+1},{i+1}]")
            lines.append("")

        result = m.T
        lines.append(f"Result: {_format_matrix(result)}")
        return "\n".join(lines)

    def _matrix_add_steps(self, expr_str: str) -> str:
        """Show step-by-step matrix addition."""
        parts = re.split(r'\]\]\s*\+\s*\[\[', expr_str)
        if len(parts) != 2:
            return "Error: Use format [[1,2],[3,4]] + [[5,6],[7,8]]"

        a_str = parts[0] + "]]"
        b_str = "[[" + parts[1]

        m1 = _parse_matrix(a_str)
        m2 = _parse_matrix(b_str)

        if m1.shape != m2.shape:
            return f"Error: Matrices must have same dimensions. Got {m1.shape[0]}x{m1.shape[1]} and {m2.shape[0]}x{m2.shape[1]}."

        rows, cols = m1.shape
        a = m1.tolist()
        b = m2.tolist()

        lines = [f"Step-by-step solution for: {_format_matrix(m1)} + {_format_matrix(m2)}", ""]
        lines.append(f"Step 1: Add corresponding elements ({rows}x{cols} matrices)")
        lines.append("")

        if rows * cols <= 16:
            step_num = 2
            for i in range(rows):
                for j in range(cols):
                    lines.append(f"  [{i+1},{j+1}]: {a[i][j]} + {b[i][j]} = {a[i][j] + b[i][j]}")
            lines.append("")

        result = m1 + m2
        lines.append(f"Result: {_format_matrix(result)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(value) -> str:
        """Format a numeric result nicely."""
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return f"{value:.10g}"
        return str(value)
