"""Session manager for saving and loading research sessions.

This module handles persisting conversation history to disk so users
can resume their research sessions later.

KEY CONCEPT: Serialization
--------------------------
We convert Python objects (the conversation history) to JSON format
so it can be saved to a file. When loading, we do the reverse.

JSON is preferred over pickle because:
1. Human-readable - you can open the file and see the content
2. Safe - pickle can execute arbitrary code when loading
3. Portable - works across Python versions

SESSION FILE STRUCTURE:
{
    "session_id": "research_20260122_143000",
    "created_at": "2026-01-22T14:30:00",
    "updated_at": "2026-01-22T15:45:00",
    "history": [
        {"input": "What is AI?", "output": "AI is..."},
        {"input": "Tell me more", "output": "..."}
    ]
}
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional


# Directory to store session files
SESSIONS_DIR = "sessions"


def ensure_sessions_dir():
    """Create the sessions directory if it doesn't exist."""
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)


def generate_session_id(description: str = None) -> str:
    """
    Generate a session ID with date and description.

    Format: dd.mm.yyyy_description_words
    Example: 22.01.2026_tesla_stock_research

    Args:
        description: Short description (3 words max). If None, uses 'session'.

    Returns:
        Session ID string (safe for filenames).
    """
    # Date in dd.mm.yyyy format
    date_str = datetime.now().strftime('%d.%m.%Y')

    if description:
        # Clean the description: lowercase, replace spaces with underscores
        # Keep only alphanumeric and underscores, limit to 3 words
        words = description.lower().split()[:3]
        clean_words = []
        for word in words:
            # Keep only alphanumeric characters
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                clean_words.append(clean_word)

        if clean_words:
            desc_str = '_'.join(clean_words)
            return f"{date_str}_{desc_str}"

    # Fallback: use timestamp to ensure uniqueness
    time_str = datetime.now().strftime('%H%M%S')
    return f"{date_str}_session_{time_str}"


def save_session(history: List[tuple], session_id: Optional[str] = None, description: Optional[str] = None) -> str:
    """
    Save the conversation history to a JSON file.

    HOW IT WORKS:
    1. Convert the list of (input, output) tuples to a list of dicts
    2. Add metadata (timestamps, session ID)
    3. Write to a JSON file

    Args:
        history: List of (user_input, agent_output) tuples from SimpleMemory
        session_id: Optional ID to use (for overwriting existing session)
        description: Short description for new sessions (3 words max)

    Returns:
        The path to the saved session file.
    """
    ensure_sessions_dir()

    # Generate or use provided session ID
    if session_id is None:
        session_id = generate_session_id(description)

    # Build the session data structure
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "message_count": len(history),
        "history": [
            {"input": inp, "output": out}
            for inp, out in history
        ]
    }

    # Save to file
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return filepath


def load_session(session_id: str) -> Optional[List[tuple]]:
    """
    Load a conversation history from a saved session file.

    Args:
        session_id: The session ID to load (filename without .json)

    Returns:
        List of (input, output) tuples, or None if session not found.
    """
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Convert back to list of tuples
        history = [
            (exchange["input"], exchange["output"])
            for exchange in session_data["history"]
        ]

        return history

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading session: {e}")
        return None


def list_sessions() -> List[Dict]:
    """
    List all saved sessions with their metadata.

    Returns:
        List of session info dicts with id, date, and message count.
    """
    ensure_sessions_dir()

    sessions = []

    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", filename[:-5]),
                        "created_at": data.get("created_at", "Unknown"),
                        "message_count": data.get("message_count", len(data.get("history", [])))
                    })
            except (json.JSONDecodeError, IOError):
                # Skip corrupted files
                continue

    # Sort by creation date (newest first)
    sessions.sort(key=lambda x: x["created_at"], reverse=True)

    return sessions


def delete_session(session_id: str) -> bool:
    """
    Delete a saved session.

    Args:
        session_id: The session ID to delete

    Returns:
        True if deleted, False if not found.
    """
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def get_session_preview(session_id: str, num_messages: int = 3) -> Optional[str]:
    """
    Get a preview of a session (first few exchanges).

    Useful for showing what a session contains before loading it.

    Args:
        session_id: The session ID
        num_messages: Number of exchanges to show (default 3)

    Returns:
        Formatted preview string, or None if session not found.
    """
    history = load_session(session_id)

    if history is None:
        return None

    if not history:
        return "Empty session (no messages)"

    preview_lines = []
    for i, (inp, out) in enumerate(history[:num_messages]):
        # Truncate long messages
        inp_preview = inp[:50] + "..." if len(inp) > 50 else inp
        out_preview = out[:50] + "..." if len(out) > 50 else out
        preview_lines.append(f"  Q: {inp_preview}")
        preview_lines.append(f"  A: {out_preview}")

    if len(history) > num_messages:
        preview_lines.append(f"  ... and {len(history) - num_messages} more exchanges")

    return "\n".join(preview_lines)
