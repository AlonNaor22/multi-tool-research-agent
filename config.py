"""Configuration settings for the Multi-Tool Research Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Module overview ───────────────────────────────────────────────
# Loads .env, exposes API keys and tuning knobs (model, temperature,
# token limits) as module-level constants.  The web UI also uses
# API_KEYS and update_env_key to render its settings panel.
# ───────────────────────────────────────────────────────────────────

# Load environment variables from .env file
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH, override=True)

# --- API key registry (used by the web UI to render the settings panel) ------
API_KEYS = {
    "ANTHROPIC_API_KEY": {
        "label": "Anthropic API Key",
        "required": True,
        "url": "https://console.anthropic.com/",
    },
    "WOLFRAM_ALPHA_APP_ID": {
        "label": "Wolfram Alpha App ID",
        "required": False,
        "url": "https://developer.wolframalpha.com/",
    },
    "OPENWEATHER_API_KEY": {
        "label": "OpenWeatherMap API Key",
        "required": False,
        "url": "https://openweathermap.org/api",
    },
    "TAVILY_API_KEY": {
        "label": "Tavily API Key",
        "required": False,
        "url": "https://tavily.com/",
    },
}


# Takes (key, value). Writes/updates the key in .env and sets os.environ.
def update_env_key(key: str, value: str) -> None:
    """Write or update a single key in the .env file and set it in os.environ."""
    os.environ[key] = value

    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    # Replace existing line or append
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"# {key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")

# REST API authentication
# When unset, the FastAPI app runs without auth (dev mode) and logs a
# prominent warning at startup. Set to any non-empty string to require
# `Authorization: Bearer <token>` on every protected endpoint.
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# LLM Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "claude-sonnet-4-5-20250929")  # Override via .env
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))  # Lower = more focused/deterministic
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))  # Maximum length of LLM response

# Agent Configuration
MAX_ITERATIONS = 10  # Maximum reasoning steps before stopping
VERBOSE = True  # Show Thought/Action/Observation steps

# Tool Configuration
SEARCH_RESULTS_LIMIT = 5  # Number of web search results to return
WIKIPEDIA_SENTENCES = 5   # Number of sentences from Wikipedia summaries
