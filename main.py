"""CLI entry point for the Multi-Tool Research Agent.

Run with: python main.py
"""

from src.agent import ResearchAgent
from src.report_generator import export_research


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("        Multi-Tool Research Agent")
    print("        Powered by Claude + LangChain")
    print("=" * 60)
    print("\nAvailable tools: web_search, wikipedia, calculator, weather, news_search, fetch_url")
    print("\nCommands:")
    print("  Type your question to research")
    print("  'clear'  - Clear conversation memory")
    print("  'quit'   - Exit the program")
    print()


def ask_to_save(query: str, answer: str):
    """Ask user if they want to save the research as a report."""
    save_input = input("Save as report? (y/n): ").strip().lower()

    if save_input in ('y', 'yes'):
        filepath = export_research(query, answer)
        print(f"Report saved to: {filepath}")


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

            # Offer to save as report
            ask_to_save(query, answer)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
