"""Main research agent using Claude and the ReAct pattern.

The agent uses Claude as a reasoning engine to decide which tools to use
and in what order, following the ReAct (Reasoning + Acting) pattern.

Now includes conversation memory for follow-up questions.
"""

from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.callbacks import TimingCallbackHandler
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


# Tool categories for hierarchical selection
# This helps the LLM navigate tools more effectively as the toolset grows
TOOL_CATEGORIES = {
    "MATH & COMPUTATION": {
        "description": "Use for calculations, equations, unit conversions, and computational knowledge.",
        "tools": ["calculator", "unit_converter", "equation_solver", "wolfram_alpha"],
        "guidance": "Use calculator for arithmetic/algebra, unit_converter for unit changes, equation_solver for symbolic math, wolfram_alpha for complex computations and verified facts."
    },
    "INFORMATION RETRIEVAL": {
        "description": "Use to find information, facts, news, and research.",
        "tools": ["web_search", "wikipedia", "news_search", "arxiv_search"],
        "guidance": "Use web_search for current events/recent data, wikipedia for established facts/history/explanations, news_search for recent news, arxiv_search for academic papers."
    },
    "WEB CONTENT": {
        "description": "Use to read and extract content from specific web pages.",
        "tools": ["fetch_url"],
        "guidance": "Use when you have a specific URL and need to read its content."
    },
    "CODE EXECUTION": {
        "description": "Use when you need to run Python code for complex logic or data processing.",
        "tools": ["python_repl"],
        "guidance": "Use for complex calculations, data manipulation, algorithms, or when other tools are insufficient."
    },
    "VISUALIZATION": {
        "description": "Use to create charts and graphs from data.",
        "tools": ["create_chart"],
        "guidance": "Use to visualize data as bar, line, or pie charts."
    },
    "MULTI-SOURCE": {
        "description": "Use to search multiple sources simultaneously.",
        "tools": ["parallel_search"],
        "guidance": "Use when you need to gather information from multiple sources at once for efficiency."
    },
    "WEATHER": {
        "description": "Use to get current weather information.",
        "tools": ["weather"],
        "guidance": "Use for weather forecasts and current conditions."
    },
}


def get_hierarchical_tool_description(tools) -> str:
    """
    Generate a hierarchical tool description string organized by category.

    This helps the LLM navigate tools more effectively by:
    1. Grouping related tools together
    2. Providing category-level guidance
    3. Giving specific tool descriptions within each category
    """
    # Build a mapping of tool names to their description
    tool_descriptions = {tool.name: tool.description for tool in tools}

    lines = []
    lines.append("Tools are organized by category. First identify the category you need, then select the appropriate tool.\n")

    for category_name, category_info in TOOL_CATEGORIES.items():
        lines.append(f"## {category_name}")
        lines.append(f"{category_info['description']}")
        lines.append(f"Guidance: {category_info['guidance']}")
        lines.append("")

        for tool_name in category_info["tools"]:
            if tool_name in tool_descriptions:
                lines.append(f"  - {tool_name}: {tool_descriptions[tool_name]}")

        lines.append("")

    return "\n".join(lines)


class SimpleMemory:
    """
    A simple conversation memory implementation.

    Stores ALL conversation exchanges for saving, but only uses the last k
    exchanges for the prompt (to avoid context overflow).

    This replaces LangChain's ConversationBufferWindowMemory which
    may not be available in all versions.
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
        # We keep ALL history now - no truncation!

    def get_history_string(self) -> str:
        """Get the recent conversation history for the prompt (last k exchanges)."""
        if not self.history:
            return "No previous conversation."

        # Only include last k exchanges in the prompt to avoid context overflow
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


# The ReAct prompt template WITH MEMORY and HIERARCHICAL TOOL SELECTION
# Notice the {chat_history} variable - this is where previous conversations go
REACT_PROMPT_WITH_MEMORY = PromptTemplate.from_template("""You are a helpful research assistant with access to various tools.
Your goal is to answer questions thoroughly by gathering information from multiple sources when needed.

Previous conversation:
{chat_history}

{tools}

TOOL SELECTION PROCESS:
1. Identify what TYPE of task you need (math? information lookup? code execution?)
2. Look at the matching CATEGORY above
3. Read the category guidance to pick the right tool
4. Choose the most specific tool for your need

Use the following format:

Question: the input question you must answer
Thought: I need to [type of task]. Looking at [CATEGORY], I should use [tool] because [reason].
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always identify the CATEGORY first, then select the tool
- Use multiple tools when necessary to gather comprehensive information
- For calculations: use calculator (simple) or python_repl (complex) - never do math in your head
- For facts: prefer wikipedia (established) over web_search (current/recent)
- For numbers/computation: prefer wolfram_alpha when precision matters
- If the user asks a follow-up question, use the chat history for context
- Synthesize information from multiple sources into a coherent answer
- If a tool returns an error, try a different approach or different tool in the same category

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


class ResearchAgent:
    """
    A research agent with conversation memory.

    This class wraps the agent so we can:
    1. Keep the same agent instance across multiple queries
    2. Maintain conversation history for follow-up questions
    3. Clear memory when needed
    """

    def __init__(self):
        """Initialize the agent with memory."""
        # Initialize Claude as the LLM
        self.llm = ChatAnthropic(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=ANTHROPIC_API_KEY,
        )

        # Collect all our tools
        self.tools = [
            calculator_tool,  # Math calculations and variables
            unit_converter_tool,  # Unit conversions (length, weight, etc.)
            equation_solver_tool,  # Solve equations (x + 2 = 5)
            wikipedia_tool,
            search_tool,
            weather_tool,
            news_tool,
            url_tool,
            arxiv_tool,  # Academic paper search
            python_repl_tool,  # Python code execution
            wolfram_tool,  # Computational knowledge (Wolfram Alpha)
            visualization_tool,  # Chart/graph generation
            parallel_tool,  # Run multiple searches in parallel
        ]

        # Create our simple conversation memory
        self.memory = SimpleMemory(k=5)

        # Track current session ID (for saving to the same file)
        self.current_session_id = None

        # Create the ReAct agent with the memory-enabled prompt
        # Use hierarchical tool descriptions for better tool selection
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT_WITH_MEMORY,
            tools_renderer=get_hierarchical_tool_description,
        )

        # Create the timing callback handler
        self.timing_callback = TimingCallbackHandler()

        # Create the executor (without built-in memory - we handle it ourselves)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=VERBOSE,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERATIONS,
        )

    def query(self, question: str, show_timing: bool = True) -> str:
        """
        Run a research query and return the answer.
        Memory is automatically updated.

        Args:
            question: The research question to answer.
            show_timing: Whether to print timing summary (default: True)

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Reset timing data from previous query
            self.timing_callback.reset()

            # Get chat history for the prompt
            chat_history = self.memory.get_history_string()

            # Run the agent with memory context
            result = self.agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": chat_history  # Pass memory to prompt
                },
                {"callbacks": [self.timing_callback]}
            )

            answer = result["output"]

            # Save this exchange to memory
            self.memory.add_exchange(question, answer)

            # Print timing summary if requested
            if show_timing:
                print(self.timing_callback.get_summary())

            return answer
        except Exception as e:
            return f"Error running research query: {str(e)}"

    def get_last_timing(self) -> str:
        """Get the timing summary from the last query."""
        return self.timing_callback.get_summary()

    def clear_memory(self):
        """Clear the conversation history and start a new session."""
        self.memory.clear()
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


# Keep backward compatibility with the old function
def create_research_agent():
    """
    Create and return a research agent (without memory).
    Kept for backward compatibility.
    """
    return ResearchAgent().agent_executor


def run_research_query(query: str) -> str:
    """
    Run a single research query (without memory).
    Kept for backward compatibility.

    For memory support, use ResearchAgent class directly.
    """
    agent = ResearchAgent()
    return agent.query(query)


# For testing
if __name__ == "__main__":
    # Test with memory
    print("Testing agent with memory...\n")
    agent = ResearchAgent()

    # First question
    q1 = "What is the population of France?"
    print(f"Q1: {q1}")
    print("=" * 60)
    a1 = agent.query(q1)
    print("=" * 60)
    print(f"A1: {a1}\n")

    # Follow-up question (uses memory)
    q2 = "How does that compare to Germany?"
    print(f"Q2: {q2}")
    print("=" * 60)
    a2 = agent.query(q2)
    print("=" * 60)
    print(f"A2: {a2}")
