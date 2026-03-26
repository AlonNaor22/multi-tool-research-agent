"""Configuration settings for the Multi-Tool Research Agent."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WOLFRAM_ALPHA_APP_ID = os.getenv("WOLFRAM_ALPHA_APP_ID")

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
