"""Main research agent using Claude and the ReAct pattern.

The agent uses Claude as a reasoning engine to decide which tools to use
and in what order, following the ReAct (Reasoning + Acting) pattern.
"""

from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

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


# The ReAct prompt template
# This tells Claude HOW to reason and use tools
REACT_PROMPT = PromptTemplate.from_template("""You are a helpful research assistant with access to various tools.
Your goal is to answer questions thoroughly by gathering information from multiple sources when needed.

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
- Synthesize information from multiple sources into a coherent answer
- If a tool returns an error, try a different approach

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


def create_research_agent():
    """
    Create and return a research agent with all tools configured.

    Returns:
        AgentExecutor: The configured agent ready to answer questions.
    """
    # Initialize Claude as the LLM (Large Language Model)
    # This is the "brain" that does the reasoning
    llm = ChatAnthropic(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=ANTHROPIC_API_KEY,
    )

    # Collect all our tools
    tools = [calculator_tool, wikipedia_tool, search_tool]

    # Create the ReAct agent
    # This combines the LLM with the prompt template and tools
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=REACT_PROMPT,
    )

    # Wrap in an executor that handles the loop
    # The executor:
    # - Runs the agent
    # - Parses its output to find Action/Action Input
    # - Calls the appropriate tool
    # - Feeds the result back as Observation
    # - Repeats until Final Answer or max_iterations
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=VERBOSE,           # Print Thought/Action/Observation steps
        handle_parsing_errors=True, # Gracefully handle malformed outputs
        max_iterations=MAX_ITERATIONS,
    )

    return agent_executor


def run_research_query(query: str) -> str:
    """
    Run a research query and return the answer.

    Args:
        query: The research question to answer.

    Returns:
        The agent's final answer as a string.
    """
    agent = create_research_agent()

    try:
        result = agent.invoke({"input": query})
        return result["output"]
    except Exception as e:
        return f"Error running research query: {str(e)}"


# For testing
if __name__ == "__main__":
    # Test with a simple query
    test_query = "What is the population of France and what percentage of the world population is that?"
    print(f"Query: {test_query}\n")
    print("=" * 60)
    answer = run_research_query(test_query)
    print("=" * 60)
    print(f"\nFinal Answer: {answer}")
