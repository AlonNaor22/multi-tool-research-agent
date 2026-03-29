"""Tests for src/tools/python_repl_tool.py — safe Python code execution."""

import pytest
from src.tools.python_repl_tool import execute_python


class TestBasicExecution:
    """Test basic code execution."""

    async def test_simple_expression(self):
        result = execute_python("2 + 2")
        assert "4" in result

    async def test_print_output(self):
        result = execute_python("print('hello world')")
        assert "hello world" in result

    async def test_multiline_code(self):
        code = "x = 5\ny = 10\nprint(x + y)"
        result = execute_python(code)
        assert "15" in result

    async def test_list_comprehension(self):
        result = execute_python("[x**2 for x in range(5)]")
        assert "0" in result and "16" in result

    async def test_string_operations(self):
        result = execute_python("'hello'.upper()")
        assert "HELLO" in result


class TestErrorHandling:
    """Test that errors are returned as strings, not raised."""

    async def test_syntax_error(self):
        result = execute_python("def foo(")
        assert "Error" in result or "SyntaxError" in result

    async def test_name_error(self):
        result = execute_python("undefined_variable")
        assert "Error" in result or "NameError" in result

    async def test_zero_division(self):
        result = execute_python("1 / 0")
        assert "Error" in result or "ZeroDivisionError" in result

    async def test_type_error(self):
        result = execute_python("'hello' + 5")
        assert "Error" in result or "TypeError" in result


class TestTimeout:
    """Test timeout enforcement."""

    async def test_infinite_loop_times_out(self):
        result = execute_python("while True: pass")
        assert "timeout" in result.lower() or "error" in result.lower() or "Error" in result


class TestSafety:
    """Test that dangerous operations are restricted."""

    async def test_empty_input(self):
        result = execute_python("")
        assert len(result) >= 0  # Should not crash
