"""Python REPL tool — process-isolated code execution with timeout and restricted builtins."""

import multiprocessing
from langchain_core.tools import tool

from src.constants import MAX_OUTPUT_LENGTH
from src.utils import truncate

# ─── Module overview ───────────────────────────────────────────────
# Runs user-supplied Python code in a sandboxed child process with
# restricted builtins, a 5-second timeout, and captured stdout/stderr.
# ───────────────────────────────────────────────────────────────────


# Configuration
EXECUTION_TIMEOUT = 5  # seconds


# Runs code inside a child process with restricted builtins and
# optional numpy/pandas. Puts (kind, value) tuple onto result_queue.
def _execute_code_in_process(code: str, result_queue: multiprocessing.Queue):
    """Run code in a child process with restricted builtins; results via queue."""
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


# Takes a code string. Spawns a child process, enforces the timeout,
# and returns captured output or an error/timeout message.
def execute_python(code: str) -> str:
    """Spawn a child process to run code, kill on timeout, return captured output."""
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


# Tool entry point. Takes Python code as a string.
# Returns execution output, truncated to MAX_OUTPUT_LENGTH.
async def python_repl(code: str) -> str:
    """Execute Python code and return the output. Use this for complex calculations, data manipulation, string processing, working with lists/dicts, or any task that requires programming logic.

    AVAILABLE MODULES: math, statistics, datetime, json, re, random, collections, itertools, functools. Also numpy (np) and pandas (pd) if installed.

    BUILT-INS: len, range, sorted, map, filter, zip, enumerate, sum, min, max, etc.

    LIMITS: 5 second timeout, 10,000 character output limit.

    EXAMPLES:
    - 'sum([1,2,3,4,5])'
    - '[x**2 for x in range(10)]'
    - 'import math; math.factorial(10)'
    - 'sorted([3,1,4,1,5,9], reverse=True)'"""
    return execute_python(code)


python_repl_tool = tool(python_repl)
