"""Persist and load conversation sessions as JSON files."""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

# ─── Module overview ───────────────────────────────────────────────
# Persists and loads conversation sessions as JSON files in the
# sessions/ directory. Each session stores (input, output) exchanges
# with metadata (timestamps, schema version, message count).
# ───────────────────────────────────────────────────────────────────

SESSIONS_DIR = "sessions"
SESSION_SCHEMA_VERSION = "1.0"


# Creates the sessions/ directory if it does not exist.
def ensure_sessions_dir():
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)


# Takes (description). Builds a filename-safe ID from the date and first 3 words.
# Returns a date+time fallback when no description is given.
def generate_session_id(description: str = None) -> str:
    """Generate a filename-safe session ID from date and optional description."""
    date_str = datetime.now().strftime('%d.%m.%Y')

    if description:
        words = description.lower().split()[:3]
        clean_words = []
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                clean_words.append(clean_word)

        if clean_words:
            desc_str = '_'.join(clean_words)
            return f"{date_str}_{desc_str}"

    time_str = datetime.now().strftime('%H%M%S')
    return f"{date_str}_session_{time_str}"


# Takes (history, session_id, description). Serializes history tuples to JSON.
# Returns the file path of the saved session.
def save_session(history: List[tuple], session_id: Optional[str] = None, description: Optional[str] = None) -> str:
    """Save history tuples to a JSON session file; return the file path."""
    ensure_sessions_dir()

    if session_id is None:
        session_id = generate_session_id(description)

    session_data = {
        "version": SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "message_count": len(history),
        "history": [
            {"input": inp, "output": out}
            for inp, out in history
        ]
    }

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return filepath


# Takes (session_id). Reads the JSON file and reconstructs (input, output) tuples.
# Returns None if the file is missing or corrupt.
def load_session(session_id: str) -> Optional[List[tuple]]:
    """Load a session by ID; return (input, output) tuples or None if missing."""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # Sessions without a version field are treated as v1.0
        version = session_data.get("version", "1.0")

        history = [
            (exchange["input"], exchange["output"])
            for exchange in session_data["history"]
        ]

        return history

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading session: {e}")
        return None


# Scans sessions/ for JSON files and returns metadata dicts sorted newest-first.
def list_sessions() -> List[Dict]:
    """Return all saved sessions as dicts with id, created_at, and message_count."""
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
                continue

    sessions.sort(key=lambda x: x["created_at"], reverse=True)

    return sessions


# Takes (session_id). Removes the session JSON file.
# Returns True if deleted, False if not found.
def delete_session(session_id: str) -> bool:
    """Delete a session file by ID; return True if found and deleted."""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


# Takes (session_id, num_messages). Loads the session and formats the first
# N exchanges as a human-readable preview string. Returns None if missing.
def get_session_preview(session_id: str, num_messages: int = 3) -> Optional[str]:
    """Return a formatted preview of the first num_messages exchanges, or None."""
    history = load_session(session_id)

    if history is None:
        return None

    if not history:
        return "Empty session (no messages)"

    preview_lines = []
    for i, (inp, out) in enumerate(history[:num_messages]):
        inp_preview = inp[:50] + "..." if len(inp) > 50 else inp
        out_preview = out[:50] + "..." if len(out) > 50 else out
        preview_lines.append(f"  Q: {inp_preview}")
        preview_lines.append(f"  A: {out_preview}")

    if len(history) > num_messages:
        preview_lines.append(f"  ... and {len(history) - num_messages} more exchanges")

    return "\n".join(preview_lines)
