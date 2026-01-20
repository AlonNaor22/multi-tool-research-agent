"""CLI entry point for the Multi-Tool Research Agent.

Run with: python main.py
"""

import sys
from src.agent import run_research_query


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("        Multi-Tool Research Agent")
    print("        Powered by Claude + LangChain")
    print("=" * 60)
    print("\nAvailable tools: web_search, wikipedia, calculator")
    print("Type 'quit' or 'exit' to stop.\n")


def main():
    """Main CLI loop."""
    print_banner()

    while True:
        try:
            # Get user input
            query = input("Your question: ").strip()

            # Check for exit commands
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            # Skip empty input
            if not query:
                continue

            print("\n" + "-" * 60)
            print("Researching...\n")

            # Run the query through the agent
            answer = run_research_query(query)

            print("-" * 60)
            print(f"\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
