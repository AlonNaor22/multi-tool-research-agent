"""Tests for src/tools/math_formatter.py — markdown rendering of math output."""

import pytest
from src.tools.math_formatter import (
    format_math_from_dict, _matrix_to_markdown, _latex_inline,
)


class TestHelpers:
    """Test helper functions."""

    def test_latex_inline(self):
        assert _latex_inline("x^2") == "$x^2$"

    def test_matrix_to_markdown(self):
        md = _matrix_to_markdown([[1, 2], [3, 4]])
        assert "|" in md
        assert "1" in md
        assert "4" in md

    def test_matrix_to_markdown_with_caption(self):
        md = _matrix_to_markdown([[1, 2]], "Test")
        assert "Test" in md

    def test_matrix_to_markdown_empty(self):
        assert _matrix_to_markdown([]) == ""


class TestFormatMathFromDict:
    """Test format_math_from_dict against representative solver payloads."""

    def test_derivative_output(self):
        data = {
            "operation": "derivative",
            "title": "d/dx(x^3)",
            "input_latex": "x^{3}",
            "steps": [
                {"num": 1, "desc": "Apply power rule", "expr_latex": "3x^{2}"},
            ],
            "result": "3*x**2",
            "result_latex": "3 x^{2}",
            "matrix_data": None,
            "result_matrix_data": None,
            "has_function": True,
            "expression_str": "x**3",
            "error": None,
        }
        result = format_math_from_dict(data)
        assert "Step 1" in result
        assert "power rule" in result
        assert "Result" in result
        assert "$" in result  # KaTeX delimiters

    def test_matrix_determinant_output(self):
        data = {
            "operation": "matrix_det",
            "title": "Determinant",
            "input_latex": "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}",
            "steps": [
                {"num": 1, "desc": "Apply 2x2 formula: ad - bc", "expr_latex": "(1)(4) - (2)(3) = -2"},
            ],
            "result": "-2",
            "result_latex": "\\det = -2",
            "matrix_data": [[1, 2], [3, 4]],
            "result_matrix_data": None,
            "has_function": False,
            "expression_str": None,
            "error": None,
        }
        result = format_math_from_dict(data)
        assert "|" in result  # matrix rendered as markdown table
        assert "Step 1" in result
        assert "Result" in result

    def test_matrix_multiply_output(self):
        data = {
            "operation": "matrix_mul",
            "title": "Matrix Multiplication",
            "input_latex": "",
            "steps": [{"num": 1, "desc": "Verify dimensions", "expr_latex": ""}],
            "result": "[[19, 22], [43, 50]]",
            "result_latex": "",
            "matrix_data": [[1, 2], [3, 4]],
            "result_matrix_data": [[19, 22], [43, 50]],
            "has_function": False,
            "expression_str": None,
            "error": None,
        }
        result = format_math_from_dict(data)
        assert "19" in result
        assert "|" in result  # markdown table

    def test_error_only(self):
        result = format_math_from_dict({"error": "Division by zero", "steps": None})
        assert "Division by zero" in result
        assert "Error" in result

    def test_singular_matrix_error_with_steps(self):
        data = {
            "operation": "matrix_inv", "title": "Inverse",
            "input_latex": "", "steps": [{"num": 1, "desc": "Det = 0", "expr_latex": ""}],
            "result": "No inverse", "result_latex": "",
            "matrix_data": [[1, 2], [2, 4]], "result_matrix_data": None,
            "has_function": False, "expression_str": None,
            "error": "Matrix is singular",
        }
        result = format_math_from_dict(data)
        assert "singular" in result.lower()
        assert "Step 1" in result

    def test_katex_delimiters_used(self):
        data = {"operation": "solve", "title": "Test", "steps": [],
                "result": "5", "result_latex": "5",
                "matrix_data": None, "result_matrix_data": None,
                "has_function": False, "expression_str": None,
                "error": None, "input_latex": ""}
        result = format_math_from_dict(data)
        assert "$" in result  # KaTeX inline delimiters


class TestStructuredSolverIntegration:
    """Test solve_structured() output works with format_math_from_dict()."""

    def test_derivative_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("derivative", "x^3 + 2x")
        assert structured.get("error") is None
        assert structured["result_latex"]
        assert len(structured["steps"]) > 0

        md = format_math_from_dict(structured)
        assert "Step 1" in md
        assert "$" in md

    def test_matrix_det_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("matrix_det", "[[3,7],[1,-4]]")
        assert structured.get("result") == str(-19)

        md = format_math_from_dict(structured)
        assert "|" in md  # markdown table
        assert "-19" in md

    def test_integral_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("integral", "x^2 from 0 to 3")
        assert structured.get("error") is None

        md = format_math_from_dict(structured)
        assert "Step" in md
        assert "9" in md

    def test_equation_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("solve", "x^2 - 4 = 0")
        assert structured.get("error") is None

        md = format_math_from_dict(structured)
        assert "Step" in md


class TestCalculatorReturnsFormattedMarkdown:
    """The calculator tool now returns ready-to-display markdown for complex ops."""

    def test_derivative_returns_markdown_not_json(self):
        from src.tools.calculator_tool import calculate
        output = calculate("derivative of x^3 + 2x")
        # No JSON envelope, no MATH_STRUCTURED prefix — just formatted markdown.
        assert "MATH_STRUCTURED" not in output
        assert not output.lstrip().startswith("{")
        # Should contain step-by-step structure and KaTeX delimiters.
        assert "Step" in output
        assert "$" in output

    def test_matrix_det_returns_markdown_table(self):
        from src.tools.calculator_tool import calculate
        output = calculate("determinant [[1,2],[3,4]]")
        assert "MATH_STRUCTURED" not in output
        assert "|" in output  # markdown table for input matrix
        assert "-2" in output  # determinant result

    def test_simple_arithmetic_unchanged(self):
        from src.tools.calculator_tool import calculate
        # Simple expressions still return plain numeric strings.
        assert calculate("2 + 2") == "4"
