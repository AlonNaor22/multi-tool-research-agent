"""Main research agent using Claude's native async tool calling.

The agent uses Claude's structured tool-use API to decide which tools to call
and in what order. All I/O is async — the agent uses ainvoke()/astream() for
non-blocking execution, the same pattern used in production agents.

Includes conversation memory for follow-up questions.
"""

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from src.callbacks import TimingCallbackHandler, StreamingCallbackHandler
from src.observability import ObservabilityCallbackHandler, MetricsStore, format_query_metrics
from src.rate_limiter import RateLimiter
from src.tool_health import check_tool_health, get_available_tools
from config import (
    ANTHROPIC_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_ITERATIONS,
    VERBOSE,
)
from src.tools.calculator_tool import calculator_tool
from src.tools.unit_converter_tool import unit_converter_tool
from src.tools.equation_solver_tool import equation_solver_tool
from src.tools.wikipedia_tool import wikipedia_tool
from src.tools.search_tool import search_tool
from src.tools.weather_tool import weather_tool
from src.tools.news_tool import news_tool
from src.tools.url_tool import url_tool
from src.tools.arxiv_tool import arxiv_tool
from src.tools.python_repl_tool import python_repl_tool
from src.tools.wolfram_tool import wolfram_tool
from src.tools.visualization_tool import visualization_tool
from src.tools.parallel_tool import parallel_tool
from src.tools.currency_tool import currency_tool
from src.tools.youtube_tool import youtube_tool
from src.tools.pdf_tool import pdf_tool
from src.tools.google_scholar_tool import google_scholar_tool
from src.tools.translation_tool import translation_tool
from src.tools.reddit_tool import reddit_tool
from src.tools.wikidata_tool import wikidata_tool


# Tool categories for hierarchical selection — included in the system prompt
# so the LLM can navigate 20 tools effectively.
TOOL_CATEGORIES = {
    "MATH & COMPUTATION": {
        "tools": ["calculator", "unit_converter", "equation_solver", "currency_converter", "wolfram_alpha"],
        "guidance": "Use calculator for arithmetic/algebra, unit_converter for unit changes, equation_solver for symbolic math, currency_converter for money exchange rates, wolfram_alpha for complex computations and verified facts."
    },
    "INFORMATION RETRIEVAL": {
        "tools": ["web_search", "wikipedia", "news_search", "arxiv_search", "youtube_search", "google_scholar"],
        "guidance": "web_search for current events/general queries, wikipedia for established facts/explanations, news_search for recent news, arxiv_search for CS/physics/math/AI pre-prints on arxiv.org, youtube_search for videos/tutorials, google_scholar for peer-reviewed papers across ALL academic fields (history, medicine, social sciences, STEM)."
    },
    "WEB CONTENT": {
        "tools": ["fetch_url", "pdf_reader"],
        "guidance": "Use fetch_url for HTML web pages, pdf_reader for PDF documents (research papers, reports)."
    },
    "SOCIAL & DISCUSSION": {
        "tools": ["reddit_search"],
        "guidance": "Use reddit_search for community opinions, discussions, experiences, and recommendations. Great for 'what do people think about X' or 'best X for Y' questions."
    },
    "KNOWLEDGE BASE": {
        "tools": ["wikidata"],
        "guidance": "Use wikidata for precise structured facts — exact dates, populations, coordinates, relationships between entities. More precise than Wikipedia for specific data points."
    },
    "TRANSLATION": {
        "tools": ["translate"],
        "guidance": "Use translate when you encounter non-English text or need to translate content for the user. Supports 100+ languages."
    },
    "CODE EXECUTION": {
        "tools": ["python_repl"],
        "guidance": "Use for complex calculations, data manipulation, algorithms, or when other tools are insufficient."
    },
    "VISUALIZATION": {
        "tools": ["create_chart"],
        "guidance": "Use to visualize data as bar, line, or pie charts."
    },
    "MULTI-SOURCE": {
        "tools": ["parallel_search"],
        "guidance": "Use when you need to gather information from multiple sources at once for efficiency."
    },
    "WEATHER": {
        "tools": ["weather"],
        "guidance": "Use for weather forecasts and current conditions."
    },
}


def _build_system_prompt(disabled_tools: list = None) -> str:
    """Build the system prompt with tool selection guidance.

    Args:
        disabled_tools: List of tool names that are unavailable (missing API keys, etc.).
                        These are removed from the category listings so the LLM doesn't try them.
    """
    disabled = set(disabled_tools or [])

    lines = [
        "You are a helpful research assistant with access to various tools.",
        "Your goal is to answer questions thoroughly by gathering information from multiple sources when needed.",
        "",
        "TOOL SELECTION PROCESS:",
        "1. Identify what TYPE of task you need (math? information lookup? code execution?)",
        "2. Look at the matching CATEGORY below",
        "3. Read the category guidance to pick the right tool",
        "4. Choose the most specific tool for your need",
        "",
    ]

    # Build categories, filtering out disabled tools
    for category_name, category_info in TOOL_CATEGORIES.items():
        available_tools = [t for t in category_info['tools'] if t not in disabled]
        if not available_tools:
            continue  # Skip entire category if all its tools are disabled
        lines.append(f"## {category_name}")
        lines.append(f"Tools: {', '.join(available_tools)}")
        lines.append(f"Guidance: {category_info['guidance']}")
        lines.append("")

    lines.extend([
        "Important guidelines:",
        "- Always identify the CATEGORY first, then select the tool",
        "- Use multiple tools when necessary to gather comprehensive information",
        "- For calculations: use calculator (simple) or python_repl (complex) - never do math in your head",
        "- For facts: prefer wikipedia (established) over web_search (current/recent)",
        "- If the user asks a follow-up question, use the conversation history for context",
        "- Synthesize information from multiple sources into a coherent answer",
        "",
        "ERROR RECOVERY:",
        "- If a tool returns an error, do NOT repeat the same call. Try an alternative tool from the same category.",
        "- If a search returns no results, try a shorter or broader query before giving up.",
        "- Fallback options: if wolfram_alpha is unavailable, use calculator or python_repl instead.",
        "- If web_search fails, try wikipedia or news_search for the same information.",
        "- Always give the user a useful answer, even if some tools are unavailable.",
    ])

    return "\n".join(lines)


# Default system prompt (no disabled tools). Overridden per-instance in __init__
# if the health check finds disabled tools.
SYSTEM_PROMPT = _build_system_prompt()


class SimpleMemory:
    """
    A simple conversation memory implementation.

    Stores ALL conversation exchanges for saving, but only uses the last k
    exchanges for the prompt (to avoid context overflow).
    """

    def __init__(self, k: int = 5):
        """
        Initialize the memory.

        Args:
            k: Number of recent exchanges to include in prompt (default: 5)
        """
        self.k = k
        self.history = []  # List of ALL (input, output) tuples

    def add_exchange(self, user_input: str, agent_output: str):
        """Add a conversation exchange to memory."""
        self.history.append((user_input, agent_output))

    def get_messages(self) -> list:
        """Get the recent conversation history as LangChain messages (last k exchanges)."""
        if not self.history:
            return []

        recent_history = self.history[-self.k:]
        messages = []
        for user_input, agent_output in recent_history:
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=agent_output))
        return messages

    def get_history_string(self) -> str:
        """Get the recent conversation history as a string (for display/compatibility)."""
        if not self.history:
            return "No previous conversation."

        recent_history = self.history[-self.k:]
        lines = []
        for user_input, agent_output in recent_history:
            lines.append(f"Human: {user_input}")
            lines.append(f"Assistant: {agent_output}")
        return "\n".join(lines)

    def clear(self):
        """Clear all conversation history."""
        self.history = []

    @property
    def buffer(self) -> str:
        """Get the memory buffer (for compatibility)."""
        return self.get_history_string()


class ResearchAgent:
    """
    A research agent using Claude's native tool calling.

    Uses LangChain's create_agent (LangGraph-based) which leverages Claude's
    structured tool-use API instead of text-based ReAct parsing. This means:
    - Tool calls are structured JSON, not fragile text parsing
    - The LLM natively understands tool schemas
    - More reliable tool selection and argument passing
    """

    def __init__(self):
        """Initialize the agent with memory and health-checked tools."""
        # Initialize Claude as the LLM
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
            streaming=True,
        )

        # Collect all our tools
        all_tools = [
            # Math & Computation
            calculator_tool,
            unit_converter_tool,
            equation_solver_tool,
            currency_tool,
            wolfram_tool,
            # Information Retrieval
            wikipedia_tool,
            search_tool,
            news_tool,
            arxiv_tool,
            youtube_tool,
            google_scholar_tool,
            # Web Content
            url_tool,
            pdf_tool,
            # Social & Discussion
            reddit_tool,
            # Knowledge Base
            wikidata_tool,
            # Translation
            translation_tool,
            # Code Execution
            python_repl_tool,
            # Visualization
            visualization_tool,
            # Multi-Source
            parallel_tool,
            # Weather
            weather_tool,
        ]

        # Run health check and filter out tools with missing dependencies.
        # Disabled tools are removed so the agent never wastes a turn calling them.
        self.tool_health = check_tool_health()
        self.tools, self.disabled_tools = get_available_tools(all_tools, self.tool_health)

        # Create our simple conversation memory
        self.memory = SimpleMemory(k=5)

        # Track current session ID (for saving to the same file)
        self.current_session_id = None

        # Create callback handlers
        self.timing_callback = TimingCallbackHandler()
        self.streaming_callback = StreamingCallbackHandler()
        self.observability_callback = ObservabilityCallbackHandler(model_name=MODEL_NAME)
        self.metrics_store = MetricsStore()
        self.rate_limiter = RateLimiter()

        # Build system prompt with disabled tools removed from categories
        system_prompt = _build_system_prompt(disabled_tools=self.disabled_tools)

        # Create the agent using native tool calling (LangGraph-based)
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            debug=VERBOSE,
        )

    async def query(self, question: str, show_timing: bool = True) -> str:
        """
        Run a research query asynchronously and return the answer.
        Memory is automatically updated.

        Args:
            question: The research question to answer.
            show_timing: Whether to print timing summary (default: True)

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Check rate limit before starting
            self.rate_limiter.check_budget()

            # Reset callbacks for new query
            self.timing_callback.reset()
            self.observability_callback.reset(question=question)

            # Build messages: conversation history + new question
            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))

            # Run the agent with native async tool calling
            result = await self.agent.ainvoke(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
            )

            # Extract the final answer from the last AI message
            answer = self._extract_answer(result)

            # Save this exchange to memory
            self.memory.add_exchange(question, answer)

            # Persist observability metrics and update rate limiter
            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            # Print timing summary if requested
            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    async def stream_query(self, question: str, show_timing: bool = True) -> str:
        """
        Run a research query with real-time async streaming output.

        Instead of blocking until the full answer is ready, this streams
        intermediate steps (thinking, tool calls) and the final answer
        token-by-token so the user sees progress in real-time.

        Args:
            question: The research question to answer.
            show_timing: Whether to print timing summary (default: True)

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Check rate limit before starting
            self.rate_limiter.check_budget()

            # Reset state from previous query
            self.timing_callback.reset()
            self.streaming_callback.reset()
            self.observability_callback.reset(question=question)

            # Build messages: conversation history + new question
            messages = self.memory.get_messages()
            messages.append(HumanMessage(content=question))

            # Stream the agent execution asynchronously
            final_result = None
            async for chunk in self.agent.astream(
                {"messages": messages},
                {"callbacks": [self.timing_callback, self.streaming_callback,
                               self.observability_callback],
                 "recursion_limit": MAX_ITERATIONS * 2},
                stream_mode="values",
            ):
                final_result = chunk

            # Extract the final answer
            answer = self._extract_answer(final_result) if final_result else "No answer was generated."

            # Save this exchange to memory
            self.memory.add_exchange(question, answer)

            # Persist observability metrics and update rate limiter
            metrics = self.observability_callback.get_metrics()
            self.metrics_store.save(metrics)
            self.rate_limiter.record_tokens(metrics.total_tokens)

            # Print timing summary if requested
            if show_timing:
                print(self.timing_callback.get_summary())
                print(format_query_metrics(metrics))

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    def _extract_answer(self, result: dict) -> str:
        """Extract the final text answer from the agent result."""
        messages = result.get("messages", [])
        # Walk backwards to find the last AI message with text content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                # Content can be a string or a list of blocks
                if isinstance(msg.content, str):
                    return msg.content
                # If it's a list of content blocks, extract text blocks
                if isinstance(msg.content, list):
                    text_parts = [
                        block["text"] for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if text_parts:
                        return "\n".join(text_parts)
        return "No answer was generated."

    def get_last_timing(self) -> str:
        """Get the timing summary from the last query."""
        return self.timing_callback.get_summary()

    def get_last_metrics(self):
        """Get the observability metrics from the last query."""
        return self.observability_callback.get_metrics()

    def clear_memory(self):
        """Clear the conversation history, reset rate limiter, and start a new session."""
        self.memory.clear()
        self.rate_limiter.reset()
        self.current_session_id = None  # Next save will create a new file
        print("Conversation memory cleared.")

    def get_memory(self) -> str:
        """Get the current conversation history."""
        return self.memory.buffer

    def save_session(self, session_id: str = None, description: str = None) -> str:
        """
        Save the current session to a file.

        Uses the same session file throughout a session. Only creates
        a new file on the first save or if explicitly given a new ID.

        Args:
            session_id: Optional session ID. If None, reuses current or generates new.
            description: Short description for new sessions (3 words max).

        Returns:
            Path to the saved session file.
        """
        from src.session_manager import save_session

        # Reuse current session ID if we have one and none was provided
        if session_id is None and self.current_session_id is not None:
            session_id = self.current_session_id

        filepath = save_session(self.memory.history, session_id, description)

        # Store the session ID for future saves (extract from filepath if new)
        if self.current_session_id is None:
            # Extract session ID from the filepath
            import os
            filename = os.path.basename(filepath)
            self.current_session_id = filename.replace('.json', '')

        return filepath

    def load_session(self, session_id: str) -> bool:
        """
        Load a previously saved session.

        Args:
            session_id: The session ID to load.

        Returns:
            True if loaded successfully, False otherwise.
        """
        from src.session_manager import load_session
        history = load_session(session_id)

        if history is None:
            return False

        # Restore the history to memory
        self.memory.history = list(history)

        # Track this as the current session (so future saves go to same file)
        self.current_session_id = session_id

        return True


def create_research_agent():
    """Create and return a research agent. Kept for backward compatibility."""
    return ResearchAgent()


async def run_research_query(query: str) -> str:
    """Run a single research query (without memory)."""
    agent = ResearchAgent()
    return await agent.query(query)


# For testing
if __name__ == "__main__":
    import asyncio

    async def _test():
        print("Testing agent with memory...\n")
        agent = ResearchAgent()

        q1 = "What is the population of France?"
        print(f"Q1: {q1}")
        print("=" * 60)
        a1 = await agent.query(q1)
        print("=" * 60)
        print(f"A1: {a1}\n")

        q2 = "How does that compare to Germany?"
        print(f"Q2: {q2}")
        print("=" * 60)
        a2 = await agent.query(q2)
        print("=" * 60)
        print(f"A2: {a2}")

    asyncio.run(_test())
