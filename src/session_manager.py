"""Session management backed by LangGraph SqliteSaver checkpoints."""

from datetime import datetime
from typing import List, Dict, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils import flatten_content

# ─── Module overview ───────────────────────────────────────────────
# Provides session listing, loading, deletion, and preview by
# querying the SqliteSaver checkpoint database. Replaces the former
# JSON-file-based session storage with a single SQLite source of truth.
# ───────────────────────────────────────────────────────────────────


# Generates a filename-safe session ID from date and optional description.
# Returns a string like "13.04.2026_climate_policy_research".
def generate_session_id(description: str = None) -> str:
    """Generate a human-readable session ID from date and optional description."""
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


# Takes (checkpointer). Queries all checkpoints grouped by thread_id.
# Returns list of dicts with session_id, created_at, message_count.
def list_sessions(checkpointer: SqliteSaver) -> List[Dict]:
    """List all conversation sessions from the checkpoint database."""
    seen = {}
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        # Skip step-scoped threads (plan-execute intermediates)
        if "__step_" in thread_id:
            continue
        if thread_id in seen:
            continue
        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        msg_count = sum(1 for m in messages if isinstance(m, (HumanMessage, AIMessage)))
        ts = checkpoint_tuple.checkpoint.get("ts", "")
        seen[thread_id] = {
            "session_id": thread_id,
            "created_at": ts or "Unknown",
            "message_count": msg_count,
        }

    sessions = list(seen.values())
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


# Takes (checkpointer, session_id). Loads checkpoint and extracts
# (user_input, agent_output) tuples for display.
def load_session(checkpointer: SqliteSaver, session_id: str) -> Optional[List[tuple]]:
    """Load a session's conversation history from the checkpoint database."""
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    if checkpoint_tuple is None:
        return None

    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    pairs = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pairs.append((msg.content, ""))
        elif isinstance(msg, AIMessage) and msg.content and pairs:
            user_input, _ = pairs[-1]
            text = flatten_content(msg.content, sep="\n") if not isinstance(msg.content, str) else msg.content
            if text:
                pairs[-1] = (user_input, text)

    return [(u, a) for u, a in pairs if a]


# Takes (checkpointer, session_id). Deletes all checkpoints for a thread.
# Returns True if checkpoints existed.
def delete_session(checkpointer: SqliteSaver, session_id: str) -> bool:
    """Delete all checkpoints for a session thread."""
    config = {"configurable": {"thread_id": session_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    if checkpoint_tuple is None:
        return False

    if hasattr(checkpointer, 'conn'):
        checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (session_id,)
        )
        checkpointer.conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", (session_id,)
        )
        checkpointer.conn.commit()
    return True


# Takes (checkpointer, session_id, num_messages). Returns a formatted
# preview of the first num_messages exchanges, or None if session missing.
def get_session_preview(checkpointer: SqliteSaver, session_id: str, num_messages: int = 3) -> Optional[str]:
    """Return a formatted preview of the first few exchanges in a session."""
    history = load_session(checkpointer, session_id)
    if history is None:
        return None
    if not history:
        return "Empty session (no messages)"

    preview_lines = []
    for user_input, agent_output in history[:num_messages]:
        inp_preview = user_input[:50] + "..." if len(user_input) > 50 else user_input
        out_preview = agent_output[:50] + "..." if len(agent_output) > 50 else agent_output
        preview_lines.append(f"  Q: {inp_preview}")
        preview_lines.append(f"  A: {out_preview}")

    if len(history) > num_messages:
        preview_lines.append(f"  ... and {len(history) - num_messages} more exchanges")

    return "\n".join(preview_lines)
