"""Python REPL tool for the research agent.

This tool allows the agent to execute Python code for complex calculations,
data manipulation, and other programmatic tasks.

SECURITY CONSIDERATIONS:
-----------------------
Executing arbitrary code is inherently risky. We implement several safeguards:

1. TIMEOUT: Code execution is limited to 5 seconds
2. PROCESS ISOLATION: Code runs in a separate process that can be killed on timeout
3. RESTRICTED GLOBALS: We limit what built-in functions are available
4. OUTPUT CAPTURE: We capture stdout/stderr instead of printing directly
5. OUTPUT SIZE LIMIT: Maximum 10,000 characters returned
6. NO PERSISTENT STATE: Each execution is independent (no shared variables)

NOTE: This is NOT a fully sandboxed environment. For production use, consider:
- Docker containers
- RestrictedPython library
- Separate subprocess with resource limits
"""

import multiprocessing
from langchain_core.tools import Tool

from src.constants import MAX_OUTPUT_LENGTH
from src.utils import truncate


# Configuration
EXECUTION_TIMEOUT = 5  # seconds


def _execute_code_in_process(code: str, result_queue: multiprocessing.Queue):
    """
    Execute code in a child process.

    Runs in a separate process so it can be killed on timeout --
    unlike threads, processes CAN be forcefully terminated.

    Args:
        code: Python code to execute
        result_queue: Queue to send the result back to the parent
    """
    import sys
    from io import StringIO
    from contextlib import redirect_stdout, redirect_stderr

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
    safe_globals.update(optional_modules)

    stdout_capture = StringIO()
    stderr_capture = StringIO()
    local_namespace = {}

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                compiled = compile(code, "<agent>", "eval")
                eval_result = eval(compiled, safe_globals, local_namespace)
                if eval_result is not None:
                    print(repr(eval_result))
            except SyntaxError:
                exec(code, safe_globals, local_namespace)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            result_queue.put(("output", f"Output:\n{output}\n\nWarnings:\n{errors}"))
        elif output:
            result_queue.put(("output", output.strip()))
        else:
            result_queue.put(("output", "Code executed successfully (no output)"))

    except Exception as e:
        error_type = type(e).__name__
        result_queue.put(("error", f"Execution Error ({error_type}): {str(e)}"))


def execute_python(code: str) -> str:
    """
    Execute Python code and return the output.

    HOW THIS WORKS:
    ---------------
    1. We spawn a child process to run the code
    2. The child captures stdout/stderr and sends results via a Queue
    3. If the process exceeds the timeout, we kill it (unlike threads, this works!)
    4. We return whatever was printed + the result of the last expression

    Args:
        code: Python code to execute (can be multi-line)

    Returns:
        The output of the code execution, or error message.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_code_in_process,
        args=(code, result_queue),
    )

    process.start()
    process.join(timeout=EXECUTION_TIMEOUT)

    # If the process is still running, kill it
    if process.is_alive():
        process.kill()    # Forcefully terminate -- no orphaned loops
        process.join(1)   # Wait briefly for cleanup
        return f"Timeout Error: Code execution exceeded {EXECUTION_TIMEOUT} seconds. The operation was too slow or contains an infinite loop."

    # Get result from the queue
    if not result_queue.empty():
        kind, value = result_queue.get_nowait()
        if kind == "error":
            return value
        output = value
    else:
        output = "Code executed successfully (no output)"

    # Truncate if too long
    output = truncate(output, MAX_OUTPUT_LENGTH, suffix=f"\n\n[Output truncated - exceeded {MAX_OUTPUT_LENGTH} characters]")

    return output


async def async_execute_python(code: str) -> str:
    """Async wrapper for the Python REPL tool."""
    return execute_python(code)


# Create the LangChain Tool wrapper
python_repl_tool = Tool(
    name="python_repl",
    func=execute_python,
    coroutine=async_execute_python,
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
