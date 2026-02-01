"""Python REPL tool for the research agent.

This tool allows the agent to execute Python code for complex calculations,
data manipulation, and other programmatic tasks.

SECURITY CONSIDERATIONS:
-----------------------
Executing arbitrary code is inherently risky. We implement several safeguards:

1. TIMEOUT: Code execution is limited to 5 seconds (works on Windows and Unix)
2. RESTRICTED GLOBALS: We limit what built-in functions are available
3. OUTPUT CAPTURE: We capture stdout/stderr instead of printing directly
4. OUTPUT SIZE LIMIT: Maximum 10,000 characters returned
5. NO PERSISTENT STATE: Each execution is independent (no shared variables)

NOTE: This is NOT a fully sandboxed environment. For production use, consider:
- Docker containers
- RestrictedPython library
- Separate subprocess with resource limits
"""

import sys
import threading
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from langchain_core.tools import Tool


# Configuration
EXECUTION_TIMEOUT = 5  # seconds
MAX_OUTPUT_LENGTH = 10000  # characters


class ExecutionResult:
    """Container for execution result from thread."""
    def __init__(self):
        self.output = None
        self.error = None


def _execute_code_thread(code: str, safe_globals: dict, result: ExecutionResult):
    """
    Execute code in a thread (allows timeout on Windows).

    Args:
        code: Python code to execute
        safe_globals: Restricted namespace
        result: ExecutionResult object to store output
    """
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    local_namespace = {}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                # First, try to compile as an expression
                compiled = compile(code, "<agent>", "eval")
                eval_result = eval(compiled, safe_globals, local_namespace)
                if eval_result is not None:
                    print(repr(eval_result))
            except SyntaxError:
                # It's not a simple expression, execute as statements
                exec(code, safe_globals, local_namespace)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            result.output = f"Output:\n{output}\n\nWarnings:\n{errors}"
        elif output:
            result.output = output.strip()
        else:
            result.output = "Code executed successfully (no output)"

    except Exception as e:
        error_type = type(e).__name__
        result.error = f"Execution Error ({error_type}): {str(e)}"


def execute_python(code: str) -> str:
    """
    Execute Python code and return the output.

    HOW THIS WORKS:
    ---------------
    1. We create a thread to run the code (allows timeout on Windows)
    2. We redirect print() output to capture it
    3. We execute the code with exec() in a restricted namespace
    4. We return whatever was printed + the result of the last expression

    Args:
        code: Python code to execute (can be multi-line)

    Returns:
        The output of the code execution, or error message.
    """
    # Import optional data science libraries if available
    optional_modules = {}

    try:
        import numpy as np
        optional_modules["np"] = np
        optional_modules["numpy"] = np
    except ImportError:
        pass

    try:
        import pandas as pd
        optional_modules["pd"] = pd
        optional_modules["pandas"] = pd
    except ImportError:
        pass

    # Define what's available to the executed code
    safe_globals = {
        "__builtins__": {
            # Safe built-ins
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "bytes": bytes,
            "chr": chr,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        },
        # Pre-import safe, useful modules
        "math": __import__("math"),
        "statistics": __import__("statistics"),
        "datetime": __import__("datetime"),
        "json": __import__("json"),
        "re": __import__("re"),
        "random": __import__("random"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
    }

    # Add optional data science modules if available
    safe_globals.update(optional_modules)

    # Create result container and execution thread
    result = ExecutionResult()
    thread = threading.Thread(
        target=_execute_code_thread,
        args=(code, safe_globals, result)
    )

    # Start thread and wait with timeout
    thread.start()
    thread.join(timeout=EXECUTION_TIMEOUT)

    # Check if thread is still running (timeout occurred)
    if thread.is_alive():
        # Note: We can't forcefully kill the thread in Python
        # The thread will continue but we return timeout error
        return f"Timeout Error: Code execution exceeded {EXECUTION_TIMEOUT} seconds. The operation was too slow or contains an infinite loop."

    # Return result
    if result.error:
        return result.error

    output = result.output or "Code executed successfully (no output)"

    # Truncate if too long
    if len(output) > MAX_OUTPUT_LENGTH:
        output = output[:MAX_OUTPUT_LENGTH] + f"\n\n[Output truncated - exceeded {MAX_OUTPUT_LENGTH} characters]"

    return output


# Create the LangChain Tool wrapper
python_repl_tool = Tool(
    name="python_repl",
    func=execute_python,
    description=(
        "Execute Python code and return the output. Use this for complex calculations, "
        "data manipulation, string processing, working with lists/dicts, or any task "
        "that requires programming logic. "
        "\n\nAVAILABLE MODULES: math, statistics, datetime, json, re, random, "
        "collections, itertools, functools. Also numpy (np) and pandas (pd) if installed."
        "\n\nBUILT-INS: len, range, sorted, map, filter, zip, enumerate, sum, min, max, etc."
        "\n\nLIMITS: 5 second timeout, 10,000 character output limit."
        "\n\nEXAMPLES:"
        "\n- 'sum([1,2,3,4,5])'"
        "\n- '[x**2 for x in range(10)]'"
        "\n- 'import math; math.factorial(10)'"
        "\n- 'sorted([3,1,4,1,5,9], reverse=True)'"
    )
)
