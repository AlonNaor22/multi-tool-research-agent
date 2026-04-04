"""Tests for src/tools/math_formatter.py — HTML rendering of math output."""

import json
import pytest
from src.tools.math_formatter import format_math, _matrix_to_html, _latex


class TestHelpers:
    """Test helper functions."""

    def test_latex_inline(self):
        assert _latex("x^2") == "\\(x^2\\)"

    def test_latex_empty(self):
        assert _latex("") == ""

    def test_matrix_to_html(self):
        html = _matrix_to_html([[1, 2], [3, 4]])
        assert "<table" in html
        assert "<td>1</td>" in html
        assert "<td>4</td>" in html

    def test_matrix_to_html_with_caption(self):
        html = _matrix_to_html([[1, 2]], "Test")
        assert "Test" in html

    def test_matrix_to_html_empty(self):
        assert _matrix_to_html([]) == ""


class TestFormatMath:
    """Test the main format_math function."""

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
        result = format_math("MATH_STRUCTURED:" + json.dumps(data))
        assert "<!-- MATH_HTML -->" in result
        assert "<!-- /MATH_HTML -->" in result
        assert "Step 1" in result
        assert "power rule" in result
        assert "MathJax" in result or "mathjax" in result
        assert "Result" in result

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
        result = format_math("MATH_STRUCTURED:" + json.dumps(data))
        assert "<!-- MATH_HTML -->" in result
        assert "<table" in result  # matrix rendered as HTML table
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
        result = format_math("MATH_STRUCTURED:" + json.dumps(data))
        assert "<!-- MATH_HTML -->" in result
        # Result matrix should be an HTML table
        assert "19" in result
        assert "<table" in result

    def test_error_only(self):
        data = {"error": "Division by zero", "steps": None}
        result = format_math(json.dumps(data))
        assert "<!-- MATH_HTML -->" in result
        assert "Division by zero" in result
        assert "math-error" in result

    def test_invalid_json(self):
        result = format_math("not valid json")
        assert "Error" in result

    def test_with_prefix(self):
        data = {"operation": "solve", "title": "Solve", "steps": [],
                "result": "x = 2", "result_latex": "x = 2",
                "matrix_data": None, "result_matrix_data": None,
                "has_function": False, "expression_str": None, "error": None,
                "input_latex": ""}
        result = format_math("MATH_STRUCTURED:" + json.dumps(data))
        assert "<!-- MATH_HTML -->" in result
        assert "x = 2" in result

    def test_function_hint(self):
        data = {
            "operation": "derivative", "title": "Test",
            "input_latex": "", "steps": [],
            "result": "2x", "result_latex": "2x",
            "matrix_data": None, "result_matrix_data": None,
            "has_function": True, "expression_str": "x**2",
            "error": None,
        }
        result = format_math(json.dumps(data))
        assert "create_chart" in result  # graph hint
        assert "x**2" in result

    def test_singular_matrix_error_with_steps(self):
        data = {
            "operation": "matrix_inv", "title": "Inverse",
            "input_latex": "", "steps": [{"num": 1, "desc": "Det = 0", "expr_latex": ""}],
            "result": "No inverse", "result_latex": "",
            "matrix_data": [[1, 2], [2, 4]], "result_matrix_data": None,
            "has_function": False, "expression_str": None,
            "error": "Matrix is singular",
        }
        result = format_math(json.dumps(data))
        assert "singular" in result.lower()
        assert "Step 1" in result

    def test_css_is_inline(self):
        data = {"operation": "solve", "title": "Test", "steps": [],
                "result": "5", "result_latex": "5",
                "matrix_data": None, "result_matrix_data": None,
                "has_function": False, "expression_str": None,
                "error": None, "input_latex": ""}
        result = format_math(json.dumps(data))
        assert "<style>" in result  # CSS is included inline


class TestStructuredSolverIntegration:
    """Test solve_structured() output works with format_math()."""

    def test_derivative_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("derivative", "x^3 + 2x")
        assert structured.get("error") is None
        assert structured["result_latex"]
        assert len(structured["steps"]) > 0

        html = format_math("MATH_STRUCTURED:" + json.dumps(structured, default=str))
        assert "<!-- MATH_HTML -->" in html
        assert "Step 1" in html

    def test_matrix_det_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("matrix_det", "[[3,7],[1,-4]]")
        assert structured.get("result") == str(-19)

        html = format_math("MATH_STRUCTURED:" + json.dumps(structured, default=str))
        assert "<table" in html
        assert "-19" in html

    def test_integral_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("integral", "x^2 from 0 to 3")
        assert structured.get("error") is None

        html = format_math("MATH_STRUCTURED:" + json.dumps(structured, default=str))
        assert "<!-- MATH_HTML -->" in html
        assert "9" in html

    def test_equation_roundtrip(self):
        from src.tools.step_solver import StepByStepSolver
        solver = StepByStepSolver()
        structured = solver.solve_structured("solve", "x^2 - 4 = 0")
        assert structured.get("error") is None

        html = format_math("MATH_STRUCTURED:" + json.dumps(structured, default=str))
        assert "<!-- MATH_HTML -->" in html
