"""Startup health checks and filtering for tools with external dependencies."""

import os
from typing import Dict, List, Tuple

# ─── Module overview ───────────────────────────────────────────────
# Startup health checks for tools with external dependencies (API
# keys, optional libraries). Probes each requirement and returns a
# health dict used to filter tools and display status in the UI.
# ───────────────────────────────────────────────────────────────────

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

LIBRARY_REQUIREMENTS = {
    "equation_solver": "sympy",
    "pdf_reader": "pdfplumber",
    "youtube_search": "yt_dlp",
    "translate": "deep_translator",
}


# Takes (env_var). Returns True if the env var is set and non-empty.
def _check_api_key(env_var: str) -> bool:
    value = os.getenv(env_var, "").strip()
    return len(value) > 0


# Takes (module_name). Returns True if the Python module can be imported.
def _check_library(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# Probes all API keys and libraries. Returns {tool_name: {available, reason, ...}}.
def check_tool_health() -> Dict[str, dict]:
    """Probe API keys and libraries; return {tool_name: {available, reason}} dict."""
    health = {}

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

    for tool_name, module_name in LIBRARY_REQUIREMENTS.items():
        if _check_library(module_name):
            health[tool_name] = {"available": True}
        else:
            fallback_map = {
                "equation_solver": "python_repl",
                "pdf_reader": "fetch_url",
                "youtube_search": "web_search with site:youtube.com",
                "translate": "web_search for translation",
            }
            health[tool_name] = {
                "available": False,
                "reason": f"{module_name} not installed",
                "fallback": fallback_map.get(tool_name, "web_search"),
            }

    return health


# Takes (all_tools, health). Filters out unhealthy tools.
# Returns (available_tools, disabled_names).
def get_available_tools(all_tools: list, health: Dict[str, dict] = None) -> Tuple[list, List[str]]:
    """Filter all_tools by health; return (available_tools, disabled_names)."""
    if health is None:
        health = check_tool_health()

    disabled_names = {
        name for name, status in health.items()
        if not status["available"]
    }

    available = [t for t in all_tools if t.name not in disabled_names]
    disabled = [t.name for t in all_tools if t.name in disabled_names]

    return available, disabled


# Takes (health, all_tool_names). Formats a human-readable status string for CLI output.
def format_health_status(health: Dict[str, dict], all_tool_names: List[str]) -> str:
    """Format tool health as a readable multi-line string for CLI."""
    lines = ["\nTool Status:"]

    always_available = [
        name for name in all_tool_names
        if name not in health
    ]
    if always_available:
        lines.append(f"  ✅ {', '.join(always_available)}")

    for tool_name, status in health.items():
        if status["available"]:
            lines.append(f"  ✅ {tool_name} (API key configured)")
        else:
            reason = status.get("reason", "unavailable")
            fallback = status.get("fallback", "")
            fallback_text = f" — using {fallback} as fallback" if fallback else ""
            lines.append(f"  ⚠️  {tool_name} ({reason}{fallback_text})")

    return "\n".join(lines)
