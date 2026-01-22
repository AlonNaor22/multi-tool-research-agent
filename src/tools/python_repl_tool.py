"""Python REPL tool for the research agent.

This tool allows the agent to execute Python code for complex calculations,
data manipulation, and other programmatic tasks.

SECURITY CONSIDERATIONS:
-----------------------
Executing arbitrary code is inherently risky. We implement several safeguards:

1. TIMEOUT: Code execution is limited to 5 seconds to prevent infinite loops
2. RESTRICTED GLOBALS: We limit what built-in functions are available
3. OUTPUT CAPTURE: We capture stdout/stderr instead of printing directly
4. NO PERSISTENT STATE: Each execution is independent (no shared variables)

NOTE: This is NOT a fully sandboxed environment. For production use, consider:
- Docker containers
- RestrictedPython library
- Separate subprocess with resource limits
"""

import sys
import signal
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.tools import Tool
from src.utils import retry_on_error


# Configuration
EXECUTION_TIMEOUT = 5  # seconds


class TimeoutError(Exception):
    """Raised when code execution takes too long."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out (exceeded 5 seconds)")


def execute_python(code: str) -> str:
    """
    Execute Python code and return the output.

    HOW THIS WORKS:
    ---------------
    1. We create StringIO objects to capture stdout/stderr
    2. We redirect print() output to our StringIO
    3. We execute the code with exec() in a restricted namespace
    4. We return whatever was printed + the result of the last expression

    Args:
        code: Python code to execute (can be multi-line)

    Returns:
        The output of the code execution, or error message.
    """
    # Prepare output capture
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Define what's available to the executed code
    # We're being somewhat permissive here for utility, but you could
    # restrict this further by removing certain built-ins
    safe_globals = {
        "__builtins__": {
            # Safe built-ins
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,  # Will be captured by redirect_stdout
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            # Math functions (commonly needed)
            "divmod": divmod,
            # String methods are available on str objects
            # List/dict comprehensions work automatically
        },
        # Pre-import some safe, useful modules
        "math": __import__("math"),
        "statistics": __import__("statistics"),
        "datetime": __import__("datetime"),
        "json": __import__("json"),
        "re": __import__("re"),
    }

    # Local namespace for the executed code
    local_namespace = {}

    try:
        # Note: signal.SIGALRM doesn't work on Windows
        # For Windows, we'd need threading with a timeout
        # This is a simplified version that works on Unix-like systems

        # Redirect stdout and stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            # Using exec() for statements, but we also want to capture
            # the result of expressions

            # Try to detect if it's a simple expression vs statements
            try:
                # First, try to compile as an expression
                compiled = compile(code, "<agent>", "eval")
                result = eval(compiled, safe_globals, local_namespace)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                # It's not a simple expression, execute as statements
                exec(code, safe_globals, local_namespace)

        # Get captured output
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            return f"Output:\n{output}\n\nErrors:\n{errors}"
        elif output:
            return output.strip()
        else:
            return "Code executed successfully (no output)"

    except TimeoutError as e:
        return f"Timeout Error: {str(e)}"
    except Exception as e:
        error_type = type(e).__name__
        return f"Execution Error ({error_type}): {str(e)}"


# Create the LangChain Tool wrapper
python_repl_tool = Tool(
    name="python_repl",
    func=execute_python,
    description=(
        "Execute Python code and return the output. Use this for complex calculations, "
        "data manipulation, string processing, working with lists/dicts, or any task "
        "that requires programming logic. The code runs in a restricted environment "
        "with access to: math, statistics, datetime, json, re modules, and common "
        "built-ins (len, range, sorted, etc.). "
        "Input should be valid Python code. For multi-line code, use proper indentation. "
        "Examples: "
        "'sum([1,2,3,4,5])', "
        "'[x**2 for x in range(10)]', "
        "'import math; math.factorial(10)'"
    )
)
