"""CLI entry point for the Multi-Tool Research Agent.

Run with: python main.py
"""

from src.agent import ResearchAgent
from src.session_manager import list_sessions, get_session_preview


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("        Multi-Tool Research Agent")
    print("        Powered by Claude + LangChain")
    print("=" * 60)
    print("\nAvailable tools: web_search, wikipedia, calculator, weather, news_search, fetch_url, arxiv_search, python_repl, wolfram_alpha, create_chart, parallel_search")
    print("\nCommands:")
    print("  Type your question to research")
    print("  'clear'    - Clear conversation memory")
    print("  'save'     - Save current session")
    print("  'load'     - Load a previous session")
    print("  'sessions' - List all saved sessions")
    print("  'quit'     - Exit the program")
    print()




def main():
    """Main CLI loop."""
    print_banner()

    # Create ONE agent instance that persists throughout the session
    # This is what enables memory - the same agent handles all queries
    print("Initializing agent with memory...")
    agent = ResearchAgent()
    print("Ready! Memory is enabled - I'll remember our conversation.\n")

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
            print("Researching...\n")

            # Run the query through the agent (memory is automatically used)
            answer = agent.query(query)

            print("-" * 60)
            print(f"\nAnswer: {answer}\n")
            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
