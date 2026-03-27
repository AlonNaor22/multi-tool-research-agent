"""Tool health checking and filtering for the research agent.

Checks which tools are available at startup based on:
- API key presence (wolfram, weather)
- Optional library availability (sympy, pypdf)

Disabled tools are filtered out before the agent sees them, so it never
wastes a turn calling a tool that will just return an error.
"""

import os
from typing import Dict, List, Tuple


# Maps tool names to their required environment variable.
# Tools not listed here are always available (no external dependency).
API_KEY_REQUIREMENTS = {
    "wolfram_alpha": {
        "env_var": "WOLFRAM_ALPHA_APP_ID",
        "setup_url": "https://developer.wolframalpha.com/",
        "fallback": "calculator or python_repl",
    },
    "weather": {
        "env_var": "OPENWEATHER_API_KEY",
        "setup_url": "https://openweathermap.org/api",
        "fallback": "web_search for weather info",
    },
}

# Maps tool names to the Python module they require.
LIBRARY_REQUIREMENTS = {
    "equation_solver": "sympy",
    "pdf_reader": "pypdf",
}


def _check_api_key(env_var: str) -> bool:
    """Check if an API key environment variable is set and non-empty."""
    value = os.getenv(env_var, "").strip()
    return len(value) > 0


def _check_library(module_name: str) -> bool:
    """Check if a Python library is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_tool_health() -> Dict[str, dict]:
    """
    Check the health of all tools that have external dependencies.

    Returns:
        Dict mapping tool name to status info:
        {
            "wolfram_alpha": {
                "available": False,
                "reason": "API key not set",
                "setup_url": "https://...",
                "fallback": "calculator or python_repl",
            },
            "weather": {"available": True},
            ...
        }

    Tools with no external dependencies are not included — they're always available.
    """
    health = {}

    # Check API-key-dependent tools
    for tool_name, req in API_KEY_REQUIREMENTS.items():
        if _check_api_key(req["env_var"]):
            health[tool_name] = {"available": True}
        else:
            health[tool_name] = {
                "available": False,
                "reason": f"{req['env_var']} not set",
                "setup_url": req["setup_url"],
                "fallback": req["fallback"],
            }

    # Check library-dependent tools
    for tool_name, module_name in LIBRARY_REQUIREMENTS.items():
        if _check_library(module_name):
            health[tool_name] = {"available": True}
        else:
            health[tool_name] = {
                "available": False,
                "reason": f"{module_name} not installed",
                "fallback": "python_repl" if tool_name == "equation_solver" else "fetch_url",
            }

    return health


def get_available_tools(all_tools: list, health: Dict[str, dict] = None) -> Tuple[list, List[str]]:
    """
    Filter the tool list, removing tools whose dependencies are missing.

    Args:
        all_tools: Full list of LangChain Tool objects.
        health: Output from check_tool_health(). If None, runs the check.

    Returns:
        Tuple of (available_tools, disabled_tool_names).
    """
    if health is None:
        health = check_tool_health()

    disabled_names = {
        name for name, status in health.items()
        if not status["available"]
    }

    available = [t for t in all_tools if t.name not in disabled_names]
    disabled = [t.name for t in all_tools if t.name in disabled_names]

    return available, disabled


def format_health_status(health: Dict[str, dict], all_tool_names: List[str]) -> str:
    """
    Format tool health into a readable startup status string.

    Args:
        health: Output from check_tool_health().
        all_tool_names: Names of all tools (for listing always-available ones).

    Returns:
        Multi-line string showing tool status.
    """
    lines = ["\nTool Status:"]

    # Always-available tools (not in health dict = no external dependency)
    always_available = [
        name for name in all_tool_names
        if name not in health
    ]
    if always_available:
        lines.append(f"  ✅ {', '.join(always_available)}")

    # Tools with dependencies
    for tool_name, status in health.items():
        if status["available"]:
            lines.append(f"  ✅ {tool_name} (API key configured)")
        else:
            reason = status.get("reason", "unavailable")
            fallback = status.get("fallback", "")
            fallback_text = f" — using {fallback} as fallback" if fallback else ""
            lines.append(f"  ⚠️  {tool_name} ({reason}{fallback_text})")

    return "\n".join(lines)
