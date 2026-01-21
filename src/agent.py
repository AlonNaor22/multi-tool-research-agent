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
from src.tools.wikipedia_tool import wikipedia_tool
from src.tools.search_tool import search_tool
from src.tools.weather_tool import weather_tool
from src.tools.news_tool import news_tool
from src.tools.url_tool import url_tool


class SimpleMemory:
    """
    A simple conversation memory implementation.

    Stores the last k conversation exchanges (question + answer pairs)
    and formats them as a string for the prompt.

    This replaces LangChain's ConversationBufferWindowMemory which
    may not be available in all versions.
    """

    def __init__(self, k: int = 5):
        """
        Initialize the memory.

        Args:
            k: Number of exchanges to remember (default: 5)
        """
        self.k = k
        self.history = []  # List of (input, output) tuples

    def add_exchange(self, user_input: str, agent_output: str):
        """Add a conversation exchange to memory."""
        self.history.append((user_input, agent_output))
        # Keep only the last k exchanges
        if len(self.history) > self.k:
            self.history = self.history[-self.k:]

    def get_history_string(self) -> str:
        """Get the conversation history as a formatted string."""
        if not self.history:
            return "No previous conversation."

        lines = []
        for user_input, agent_output in self.history:
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


# The ReAct prompt template WITH MEMORY
# Notice the {chat_history} variable - this is where previous conversations go
REACT_PROMPT_WITH_MEMORY = PromptTemplate.from_template("""You are a helpful research assistant with access to various tools.
Your goal is to answer questions thoroughly by gathering information from multiple sources when needed.

Previous conversation:
{chat_history}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always think step by step about what information you need
- Use multiple tools when necessary to gather comprehensive information
- Use the calculator for any mathematical calculations - don't do math in your head
- Use wikipedia for background/encyclopedic information
- Use web_search for current events, recent data, or real-time information
- If the user asks a follow-up question, use the chat history for context
- Synthesize information from multiple sources into a coherent answer
- If a tool returns an error, try a different approach

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
            calculator_tool,
            wikipedia_tool,
            search_tool,
            weather_tool,
            news_tool,
            url_tool
        ]

        # Create our simple conversation memory
        self.memory = SimpleMemory(k=5)

        # Create the ReAct agent with the memory-enabled prompt
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT_WITH_MEMORY,
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
        """Clear the conversation history."""
        self.memory.clear()
        print("Conversation memory cleared.")

    def get_memory(self) -> str:
        """Get the current conversation history."""
        return self.memory.buffer


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
