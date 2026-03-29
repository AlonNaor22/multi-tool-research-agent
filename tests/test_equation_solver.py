"""Tests for src/tools/equation_solver_tool.py — equations, systems, matrices, calculus."""

import pytest
from src.tools.equation_solver_tool import solve_equation


class TestLinearEquations:
    """Test solving linear equations."""

    async def test_simple_linear(self):
        result = solve_equation("2*x + 4 = 10")
        assert "3" in result

    async def test_negative_solution(self):
        result = solve_equation("x + 5 = 2")
        assert "-3" in result

    async def test_fractional_solution(self):
        result = solve_equation("3*x = 1")
        assert "1/3" in result or "0.33" in result


class TestQuadraticEquations:
    """Test solving quadratic equations."""

    async def test_simple_quadratic(self):
        result = solve_equation("x^2 - 4 = 0")
        assert "-2" in result and "2" in result

    async def test_quadratic_single_root(self):
        result = solve_equation("x^2 = 0")
        assert "0" in result


class TestPreprocessing:
    """Test expression preprocessing."""

    async def test_implicit_multiplication(self):
        result = solve_equation("2x + 4 = 10")
        assert "3" in result

    async def test_power_caret_notation(self):
        result = solve_equation("x^2 = 9")
        assert "3" in result or "-3" in result


class TestSystemsOfEquations:
    """Test solving systems of equations."""

    async def test_two_variable_system(self):
        result = solve_equation("system: x + y = 5, 2x - y = 1")
        assert "2" in result
        assert "3" in result

    async def test_three_variable_system(self):
        result = solve_equation("system: a + b + c = 6, a - b = 0, b + c = 4")
        assert "Solution" in result

    async def test_system_no_solution(self):
        # Parallel lines: x + y = 1, x + y = 3
        result = solve_equation("system: x + y = 1, x + y = 3")
        assert "No solution" in result or "no" in result.lower()

    async def test_system_single_equation_error(self):
        result = solve_equation("system: x + y = 5")
        assert "Error" in result


class TestMatrixOperations:
    """Test matrix operations."""

    async def test_matrix_multiply(self):
        result = solve_equation("matrix: [[1,2],[3,4]] * [[5,6],[7,8]]")
        assert "19" in result  # 1*5 + 2*7 = 19
        assert "22" in result  # 1*6 + 2*8 = 22

    async def test_inverse(self):
        result = solve_equation("inverse: [[1,2],[3,4]]")
        assert "Inverse" in result

    async def test_inverse_singular(self):
        result = solve_equation("inverse: [[1,2],[2,4]]")
        assert "singular" in result.lower() or "Error" in result

    async def test_determinant(self):
        result = solve_equation("det: [[1,2],[3,4]]")
        assert "-2" in result  # 1*4 - 2*3 = -2

    async def test_eigenvalues(self):
        result = solve_equation("eigenvalues: [[2,0],[0,3]]")
        assert "2" in result
        assert "3" in result

    async def test_transpose(self):
        result = solve_equation("transpose: [[1,2,3],[4,5,6]]")
        assert "Transpose" in result

    async def test_rank(self):
        result = solve_equation("rank: [[1,0],[0,1]]")
        assert "2" in result

    async def test_rref(self):
        result = solve_equation("rref: [[1,2,3],[4,5,6]]")
        assert "echelon" in result.lower()


class TestSymbolicAlgebra:
    """Test symbolic algebra operations."""

    async def test_simplify(self):
        result = solve_equation("simplify: (x^2 - 1)/(x - 1)")
        assert "x + 1" in result or "1 + x" in result

    async def test_expand(self):
        result = solve_equation("expand: (x + 1)^2")
        assert "x**2" in result or "x^2" in result
        assert "2*x" in result or "2x" in result

    async def test_factor(self):
        result = solve_equation("factor: x^2 + 5x + 6")
        assert "x + 2" in result
        assert "x + 3" in result

    async def test_derivative(self):
        result = solve_equation("derivative: x^3 + 2x")
        assert "3*x**2" in result or "3x^2" in result
        assert "2" in result

    async def test_integral(self):
        result = solve_equation("integrate: x^2")
        assert "x**3" in result or "x^3" in result
        assert "3" in result  # x^3/3
        assert "+ C" in result

    async def test_diff_shorthand(self):
        result = solve_equation("diff: sin(x)")
        assert "cos" in result


class TestErrorHandling:
    """Test error cases."""

    async def test_empty_input(self):
        result = solve_equation("")
        assert "Error" in result

    async def test_no_equals_sign(self):
        result = solve_equation("x + 5")
        assert len(result) > 0  # Solves as x + 5 = 0

    async def test_help_command(self):
        result = solve_equation("help")
        assert "SYSTEMS" in result
        assert "MATRIX" in result
        assert "CALCULUS" in result

    async def test_invalid_matrix(self):
        result = solve_equation("inverse: not_a_matrix")
        assert "Error" in result or "Cannot parse" in result
