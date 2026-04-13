"""CLI entry point for the Multi-Tool Research Agent.

Run with:
  python main.py              # direct mode (default)
  python main.py --plan       # plan-and-execute mode
"""

import argparse
import json
import os
import re
import sys
import asyncio
from src.agent import ResearchAgent
from src.session_manager import list_sessions, get_session_preview
from src.tool_health import format_health_status
from src.observability import MetricsStore
from src.utils import close_aiohttp_session

# ─── Module overview ───────────────────────────────────────────────
# CLI entry point.  Parses --plan / --multi-agent flags, boots the
# agent, and runs an interactive REPL with session save/load,
# memory clear, stats display, and clean math output for terminals.
# ───────────────────────────────────────────────────────────────────


# Takes raw agent text. Replaces MATH_STRUCTURED: JSON and HTML math
# blocks with plain-text fallbacks safe for terminal display.
# Returns cleaned string.
def _clean_math_output(text: str) -> str:
    """Clean math output for CLI display.

    - Replaces raw MATH_STRUCTURED: JSON with plain-text fallback
    - Strips HTML sentinel blocks (can't render in terminal)

    This is the safety net for direct mode when the agent passes structured
    output through without calling math_formatter, or returns HTML that
    can't render in a terminal.
    """
    # Strip HTML math blocks — extract plain text from them if possible
    if '<!-- MATH_HTML -->' in text:
        text = re.sub(
            r'<!-- MATH_HTML -->.*?<!-- /MATH_HTML -->',
            '[Math output rendered in Web UI]',
            text,
            flags=re.DOTALL,
        )

    if 'MATH_STRUCTURED:' not in text:
        return text

    result_parts = []
    remaining = text

    while 'MATH_STRUCTURED:' in remaining:
        prefix = 'MATH_STRUCTURED:'
        idx = remaining.index(prefix)
        result_parts.append(remaining[:idx])

        json_start = idx + len(prefix)
        if json_start >= len(remaining) or remaining[json_start] != '{':
            result_parts.append(remaining[idx:])
            remaining = ""
            break

        # Find matching closing brace
        depth = 0
        json_end = json_start
        for i in range(json_start, len(remaining)):
            if remaining[i] == '{':
                depth += 1
            elif remaining[i] == '}':
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        try:
            data = json.loads(remaining[json_start:json_end])
            plain = data.get("plain_text", remaining[idx:json_end])
            result_parts.append(plain)
        except (json.JSONDecodeError, Exception):
            result_parts.append(remaining[idx:json_end])

        remaining = remaining[json_end:]

    result_parts.append(remaining)
    return "".join(result_parts)


# Parses CLI flags (--plan, --multi-agent).
# Returns argparse.Namespace with boolean flags.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Tool Research Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py              # direct mode (default)\n"
            "  python main.py --plan       # always use plan-and-execute\n"
            "  python main.py --multi-agent # multi-agent orchestration\n"
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Force plan-and-execute mode for every query.",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        default=False,
        help="Use multi-agent orchestration (supervisor delegates to specialists).",
    )
    return parser.parse_args()


# Takes (plan_mode, multi_agent_mode). Prints the CLI welcome banner
# with available commands and the active execution mode.
def print_banner(plan_mode: bool = False, multi_agent_mode: bool = False):
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("        Multi-Tool Research Agent")
    print("        Powered by Claude + LangChain (Native Tool Calling)")
    if multi_agent_mode:
        print("        Mode: Multi-Agent Orchestration")
    elif plan_mode:
        print("        Mode: Plan-and-Execute")
    print("=" * 60)
    print("\nCommands:")
    print("  Type your question to research")
    print("  'clear'    - Clear conversation memory")
    print("  'load'     - Load a previous session")
    print("  'sessions' - List all saved sessions")
    print("  'history'  - Show conversation history")
    print("  'stats'    - Show performance stats")
    print("  'quit'     - Exit the program")
    print()


# Main async REPL loop.  Initialises the agent, checks API keys,
# then dispatches each query via direct / plan / multi-agent mode.
async def main():
    """Main async CLI loop."""
    args = parse_args()
    plan_mode = args.plan
    multi_agent_mode = args.multi_agent

    print_banner(plan_mode=plan_mode, multi_agent_mode=multi_agent_mode)

    # Fail fast if the required Anthropic API key is missing
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("Add it to your .env file or export it as an environment variable.")
        print("Get a key at: https://console.anthropic.com/")
        sys.exit(1)

    # Create ONE agent instance that persists throughout the session
    # This is what enables memory - the same agent handles all queries
    print("Initializing agent...")
    agent = ResearchAgent()

    # Show tool health status — which tools are available vs disabled
    all_tool_names = [t.name for t in agent.tools]
    print(format_health_status(agent.tool_health, all_tool_names))

    if agent.disabled_tools:
        print(f"\n{len(agent.tools)} tools active, {len(agent.disabled_tools)} disabled.")
    else:
        print(f"\nAll {len(agent.tools)} tools active.")

    print("Memory is enabled - I'll remember our conversation.\n")

    while True:
        try:
            # Get user input
            query = input("Your question: ").strip()

            # Check for exit commands
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            # Check for clear memory command
            if query.lower() == "clear":
                agent.clear_memory()
                print("Started fresh conversation.\n")
                continue

            # Save is no longer needed — sessions are auto-persisted
            if query.lower() == "save":
                print("Sessions are auto-saved.\n")
                continue

            # Check for load session command
            if query.lower() == "load":
                sessions = list_sessions(agent.checkpointer)
                if not sessions:
                    print("No saved sessions found.\n")
                else:
                    print("\nAvailable sessions:")
                    for i, s in enumerate(sessions, 1):
                        print(f"  {i}. {s['session_id']} ({s['message_count']} messages)")

                    choice = input("\nEnter session number or ID (or 'cancel'): ").strip()

                    if choice.lower() == 'cancel':
                        print("Cancelled.\n")
                    else:
                        # Handle numeric choice
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(sessions):
                                session_id = sessions[idx]['session_id']
                            else:
                                print("Invalid number.\n")
                                continue
                        except ValueError:
                            session_id = choice  # Treat as session ID

                        if agent.load_session(session_id):
                            print(f"Loaded session: {session_id}\n")
                        else:
                            print(f"Could not load session: {session_id}\n")
                continue

            # Check for stats command
            if query.lower() == "stats":
                store = MetricsStore()
                print(store.format_summary())
                print()
                continue

            # Check for list sessions command
            if query.lower() == "sessions":
                sessions = list_sessions(agent.checkpointer)
                if not sessions:
                    print("No saved sessions found.\n")
                else:
                    print("\nSaved sessions:")
                    print("-" * 50)
                    for s in sessions:
                        print(f"  {s['session_id']}")
                        print(f"    Created: {s['created_at'][:19]}")
                        print(f"    Messages: {s['message_count']}")
                        preview = get_session_preview(agent.checkpointer, s['session_id'], 1)
                        if preview:
                            print(f"    Preview:\n{preview}")
                        print()
                continue

            # Check for history command
            if query.lower() == "history":
                history = agent.get_conversation_history()
                if not history:
                    print("No conversation history yet.\n")
                else:
                    print("\nConversation history:")
                    print("-" * 50)
                    for i, (user_input, assistant_output) in enumerate(history, 1):
                        print(f"  [{i}] You: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")
                        print(f"      Agent: {assistant_output[:80]}{'...' if len(assistant_output) > 80 else ''}")
                    print()
                continue

            # Skip empty input
            if not query:
                continue

            print("\n" + "-" * 60)

            if multi_agent_mode:
                # Multi-agent mode: supervisor delegates to specialist agents
                answer = await agent.multi_agent_query(query)
            elif plan_mode:
                # Plan-and-execute mode: generates a structured plan first,
                # executes each step, then synthesizes findings.
                answer = await agent.plan_and_execute(query)
            else:
                # Direct mode: async streaming — shows thinking/tool use in real-time
                answer = await agent.stream_query(query)

            print("\n" + "-" * 60)
            if not plan_mode and not multi_agent_mode:
                # In plan/multi-agent mode the answer is already streamed to stdout
                # Clean any raw MATH_STRUCTURED: JSON for terminal display
                print(f"\nAnswer: {_clean_math_output(answer)}\n")
            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Ensure the shared aiohttp session is closed
        try:
            asyncio.run(close_aiohttp_session())
        except RuntimeError:
            pass
