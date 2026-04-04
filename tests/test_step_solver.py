"""Tests for src/tools/step_solver.py — step-by-step math solutions."""

import pytest
from src.tools.step_solver import StepByStepSolver, detect_operation


@pytest.fixture
def solver():
    return StepByStepSolver()


# ============================================================================
# OPERATION DETECTION
# ============================================================================

class TestDetectOperation:
    """Test the operation classifier."""

    def test_simple_arithmetic(self):
        op, _ = detect_operation("2 + 2")
        assert op == "simple"

    def test_simple_function(self):
        op, _ = detect_operation("sqrt(16)")
        assert op == "simple"

    def test_complex_arithmetic(self):
        op, _ = detect_operation("(2+3)*4 - 10/2")
        assert op == "complex_arithmetic"

    def test_deeply_nested(self):
        op, _ = detect_operation("((2+3)*4)")
        assert op == "complex_arithmetic"

    def test_derivative_keyword(self):
        op, expr = detect_operation("derivative of x^3")
        assert op == "derivative"
        assert "x^3" in expr

    def test_diff_keyword(self):
        op, _ = detect_operation("diff x^2 + 3x")
        assert op == "derivative"

    def test_d_dx_keyword(self):
        op, _ = detect_operation("d/dx x^3 + 2x")
        assert op == "derivative"

    def test_integrate_keyword(self):
        op, expr = detect_operation("integrate x^2 from 0 to 5")
        assert op == "integral"

    def test_integral_keyword(self):
        op, _ = detect_operation("integral of 3x + 1")
        assert op == "integral"

    def test_solve_keyword(self):
        op, _ = detect_operation("solve x^2 - 4 = 0")
        assert op == "solve"

    def test_equation_with_equals(self):
        op, _ = detect_operation("2x + 3 = 7")
        assert op == "solve"

    def test_matrix_determinant(self):
        op, _ = detect_operation("determinant [[3,7],[1,-4]]")
        assert op == "matrix_det"

    def test_matrix_det_short(self):
        op, _ = detect_operation("det [[1,2],[3,4]]")
        assert op == "matrix_det"

    def test_matrix_multiply(self):
        op, _ = detect_operation("[[1,2],[3,4]] * [[5,6],[7,8]]")
        assert op == "matrix_mul"

    def test_matrix_inverse(self):
        op, _ = detect_operation("inverse [[1,2],[3,4]]")
        assert op == "matrix_inv"

    def test_matrix_transpose(self):
        op, _ = detect_operation("transpose [[1,2,3],[4,5,6]]")
        assert op == "matrix_trans"

    def test_matrix_add(self):
        op, _ = detect_operation("[[1,2],[3,4]] + [[5,6],[7,8]]")
        assert op == "matrix_add"

    def test_passthrough_variables(self):
        op, _ = detect_operation("variables")
        assert op == "passthrough"

    def test_passthrough_clear(self):
        op, _ = detect_operation("clear")
        assert op == "passthrough"

    def test_passthrough_help(self):
        op, _ = detect_operation("help")
        assert op == "passthrough"

    def test_passthrough_assignment(self):
        op, _ = detect_operation("x = 10")
        assert op == "passthrough"

    def test_passthrough_set_assignment(self):
        op, _ = detect_operation("set y = 25")
        assert op == "passthrough"


# ============================================================================
# ARITHMETIC STEPS
# ============================================================================

class TestArithmeticSteps:
    """Test step-by-step arithmetic breakdowns."""

    def test_basic_complex(self, solver):
        result = solver.solve("complex_arithmetic", "(2+3)*4")
        assert "Step" in result
        assert "Result:" in result
        assert "20" in result

    def test_shows_multiple_steps(self, solver):
        result = solver.solve("complex_arithmetic", "(3+4)*(5-2)/(1+2)")
        assert "Step 1" in result
        assert "Step 2" in result
        assert "Result:" in result
        assert "7" in result  # final result is 7

    def test_order_of_operations(self, solver):
        result = solver.solve("complex_arithmetic", "2 + 3 * 4")
        assert "Result:" in result
        assert "14" in result

    def test_invalid_syntax(self, solver):
        result = solver.solve("complex_arithmetic", "2 +* 3")
        assert "Error" in result


# ============================================================================
# DERIVATIVE STEPS
# ============================================================================

class TestDerivativeSteps:
    """Test step-by-step differentiation."""

    def test_power_rule(self, solver):
        result = solver.solve("derivative", "x^3")
        assert "Step" in result
        assert "power rule" in result.lower()
        assert "Result:" in result
        assert "3*x**2" in result or "3x^2" in result or "3*x^2" in result

    def test_multiple_terms(self, solver):
        result = solver.solve("derivative", "x^2 + 3x - 5")
        assert "sum rule" in result.lower() or "term" in result.lower()
        assert "Result:" in result

    def test_trig_derivative(self, solver):
        result = solver.solve("derivative", "sin(x)")
        assert "cos" in result
        assert "Result:" in result

    def test_constant_derivative(self, solver):
        result = solver.solve("derivative", "5 + x - x")
        assert "0" in result or "Result:" in result

    def test_no_variable_error(self, solver):
        result = solver.solve("derivative", "")
        assert "Error" in result

    def test_no_variable_constant_only(self, solver):
        result = solver.solve("derivative", "5")
        assert "Error" in result  # no variable to differentiate with respect to


# ============================================================================
# INTEGRAL STEPS
# ============================================================================

class TestIntegralSteps:
    """Test step-by-step integration."""

    def test_power_rule_integral(self, solver):
        result = solver.solve("integral", "x^2")
        assert "Step" in result
        assert "Result:" in result
        assert "x**3/3" in result or "x^3/3" in result

    def test_definite_integral(self, solver):
        result = solver.solve("integral", "x^2 from 0 to 3")
        assert "F(" in result  # evaluation step
        assert "Result:" in result
        assert "9" in result  # 3^3/3 = 9

    def test_multiple_terms_integral(self, solver):
        result = solver.solve("integral", "3x + 1")
        assert "sum rule" in result.lower() or "term" in result.lower()
        assert "Result:" in result
        assert "+ C" in result

    def test_no_variable_error(self, solver):
        result = solver.solve("integral", "")
        assert "Error" in result


# ============================================================================
# EQUATION SOLVING STEPS
# ============================================================================

class TestEquationSteps:
    """Test step-by-step equation solving."""

    def test_linear_equation(self, solver):
        result = solver.solve("solve", "2x + 3 = 7")
        assert "Step" in result
        assert "Result:" in result
        assert "2" in result

    def test_quadratic_equation(self, solver):
        result = solver.solve("solve", "x^2 - 4 = 0")
        assert "discriminant" in result.lower()
        assert "Result:" in result
        assert "2" in result and "-2" in result

    def test_quadratic_with_formula(self, solver):
        result = solver.solve("solve", "x^2 + 5x + 6 = 0")
        assert "Result:" in result
        assert "-2" in result and "-3" in result

    def test_no_variable_error(self, solver):
        result = solver.solve("solve", "5 = 5")
        assert "Error" in result or "No" in result


# ============================================================================
# MATRIX STEPS
# ============================================================================

class TestMatrixDeterminantSteps:
    """Test step-by-step determinant calculation."""

    def test_2x2_determinant(self, solver):
        result = solver.solve("matrix_det", "[[3,7],[1,-4]]")
        assert "ad - bc" in result.lower() or "ad" in result.lower()
        assert "Result:" in result
        assert "-19" in result

    def test_3x3_determinant(self, solver):
        result = solver.solve("matrix_det", "[[1,2,3],[4,5,6],[7,8,9]]")
        assert "cofactor" in result.lower()
        assert "Result:" in result
        assert "0" in result  # singular matrix

    def test_non_square_error(self, solver):
        result = solver.solve("matrix_det", "[[1,2,3],[4,5,6]]")
        assert "Error" in result


class TestMatrixMultiplySteps:
    """Test step-by-step matrix multiplication."""

    def test_2x2_multiply(self, solver):
        result = solver.solve("matrix_mul", "[[1,2],[3,4]] * [[5,6],[7,8]]")
        assert "dot product" in result.lower() or "Row" in result
        assert "Result:" in result
        assert "19" in result  # element [1,1]

    def test_dimension_mismatch(self, solver):
        result = solver.solve("matrix_mul", "[[1,2,3]] * [[4,5]]")
        assert "Error" in result


class TestMatrixInverseSteps:
    """Test step-by-step matrix inversion."""

    def test_2x2_inverse(self, solver):
        result = solver.solve("matrix_inv", "[[1,2],[3,4]]")
        assert "determinant" in result.lower()
        assert "Result:" in result

    def test_singular_matrix(self, solver):
        result = solver.solve("matrix_inv", "[[1,2],[2,4]]")
        assert "singular" in result.lower() or "not invertible" in result.lower()


class TestMatrixTransposeSteps:
    """Test step-by-step matrix transposition."""

    def test_transpose(self, solver):
        result = solver.solve("matrix_trans", "[[1,2,3],[4,5,6]]")
        assert "rows and columns" in result.lower() or "Row" in result
        assert "Result:" in result


class TestMatrixAddSteps:
    """Test step-by-step matrix addition."""

    def test_addition(self, solver):
        result = solver.solve("matrix_add", "[[1,2],[3,4]] + [[5,6],[7,8]]")
        assert "Result:" in result
        assert "6" in result  # 1+5

    def test_dimension_mismatch(self, solver):
        result = solver.solve("matrix_add", "[[1,2],[3,4]] + [[5,6,7]]")
        assert "Error" in result


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_operation(self, solver):
        result = solver.solve("unknown_op", "something")
        assert "Error" in result

    def test_empty_derivative(self, solver):
        result = solver.solve("derivative", "")
        assert "Error" in result

    def test_empty_integral(self, solver):
        result = solver.solve("integral", "")
        assert "Error" in result
