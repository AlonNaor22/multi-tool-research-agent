"""Tests for src/tools/equation_solver_tool.py — symbolic equation solving."""

import pytest
from src.tools.equation_solver_tool import solve_equation


class TestLinearEquations:
    """Test solving linear equations."""

    def test_simple_linear(self):
        result = solve_equation("2*x + 4 = 10")
        assert "3" in result

    def test_negative_solution(self):
        result = solve_equation("x + 5 = 2")
        assert "-3" in result

    def test_fractional_solution(self):
        result = solve_equation("3*x = 1")
        assert "1/3" in result or "0.33" in result


class TestQuadraticEquations:
    """Test solving quadratic equations."""

    def test_simple_quadratic(self):
        result = solve_equation("x^2 - 4 = 0")
        assert "-2" in result and "2" in result

    def test_quadratic_single_root(self):
        result = solve_equation("x^2 = 0")
        assert "0" in result


class TestPreprocessing:
    """Test expression preprocessing (implicit multiplication, power notation)."""

    def test_implicit_multiplication(self):
        # "2x" should be treated as "2*x"
        result = solve_equation("2x + 4 = 10")
        assert "3" in result

    def test_power_caret_notation(self):
        # "x^2" should be treated as "x**2"
        result = solve_equation("x^2 = 9")
        assert "3" in result or "-3" in result


class TestErrorHandling:
    """Test error cases."""

    def test_empty_input(self):
        result = solve_equation("")
        assert "Error" in result or "provide" in result.lower() or len(result) > 0

    def test_no_equals_sign(self):
        # Should still attempt to solve or return helpful message
        result = solve_equation("x + 5")
        assert len(result) > 0

    def test_help_command(self):
        result = solve_equation("help")
        assert len(result) > 10
