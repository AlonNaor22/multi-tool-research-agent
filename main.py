"""CLI entry point for the Multi-Tool Research Agent.

Run with: python main.py
"""

import os
import sys
import asyncio
from src.agent import ResearchAgent
from src.session_manager import list_sessions, get_session_preview
from src.tool_health import format_health_status
from src.observability import MetricsStore
from src.utils import close_aiohttp_session


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("        Multi-Tool Research Agent")
    print("        Powered by Claude + LangChain (Native Tool Calling)")
    print("=" * 60)
    print("\nCommands:")
    print("  Type your question to research")
    print("  'clear'    - Clear conversation memory")
    print("  'save'     - Save current session")
    print("  'load'     - Load a previous session")
    print("  'sessions' - List all saved sessions")
    print("  'stats'    - Show performance stats")
    print("  'quit'     - Exit the program")
    print()


async def main():
    """Main async CLI loop."""
    print_banner()

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

            # Check for save session command
            if query.lower() == "save":
                if not agent.memory.history:
                    print("Nothing to save - no conversation yet.\n")
                else:
                    is_new = agent.current_session_id is None
                    description = None

                    # Ask for description only when creating a new session
                    if is_new:
                        description = input("Session description (3 words max, or press Enter to skip): ").strip()
                        if not description:
                            description = None

                    filepath = agent.save_session(description=description)
                    if is_new:
                        print(f"New session created: {filepath}\n")
                    else:
                        print(f"Session updated: {filepath}\n")
                continue

            # Check for load session command
            if query.lower() == "load":
                sessions = list_sessions()
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
                            print(f"Loaded session: {session_id}")
                            print(f"Restored {len(agent.memory.history)} messages.\n")
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
                sessions = list_sessions()
                if not sessions:
                    print("No saved sessions found.\n")
                else:
                    print("\nSaved sessions:")
                    print("-" * 50)
                    for s in sessions:
                        print(f"  {s['session_id']}")
                        print(f"    Created: {s['created_at'][:19]}")
                        print(f"    Messages: {s['message_count']}")
                        preview = get_session_preview(s['session_id'], 1)
                        if preview:
                            print(f"    Preview:\n{preview}")
                        print()
                continue

            # Skip empty input
            if not query:
                continue

            print("\n" + "-" * 60)

            # Run the query with async streaming — shows thinking/tool use in real-time
            answer = await agent.stream_query(query)

            print("\n" + "-" * 60)
            print(f"\nAnswer: {answer}\n")
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
