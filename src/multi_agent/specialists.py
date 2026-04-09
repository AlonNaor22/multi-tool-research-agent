"""Specialist agents for the multi-agent orchestration system.

Each specialist wraps a LangGraph tool-calling agent with a focused
tool subset and system prompt. The supervisor delegates tasks to these
specialists, which run independently (and potentially in parallel).
"""

import asyncio
from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

from src.tool_health import get_available_tools
from src.multi_agent.prompts import (
    RESEARCH_AGENT_PROMPT,
    MATH_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    FACT_CHECKER_PROMPT,
    TRANSLATION_AGENT_PROMPT,
)


DEFAULT_RECURSION_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 120.0


# Per-specialist configuration.
#
# ``recursion_limit`` caps how many tool-call loops the inner LangGraph
# agent may take before raising. Research-heavy roles get more headroom;
# narrow roles (translation, fact_checker) get less so they fail fast.
#
# ``timeout_seconds`` bounds wall-clock time per specialist invocation.
# On timeout, the specialist returns a degraded string and the phase
# continues — one stuck role no longer blocks the whole phase.
SPECIALIST_DEFINITIONS = {
    "research": {
        "tools": [
            "web_search", "wikipedia", "news_search", "arxiv_search",
            "google_scholar", "reddit_search", "youtube_search",
            "fetch_url", "pdf_reader", "wikidata", "github_search",
            "web_scraper", "parallel_search",
        ],
        "prompt": RESEARCH_AGENT_PROMPT,
        "recursion_limit": 25,
        "timeout_seconds": 180.0,
    },
    "math": {
        "tools": [
            "calculator", "unit_converter", "equation_solver",
            "currency_converter", "wolfram_alpha", "python_repl",
            "datetime_calculator", "math_formatter", "create_chart",
        ],
        "prompt": MATH_AGENT_PROMPT,
        "recursion_limit": 15,
        "timeout_seconds": 90.0,
    },
    "analysis": {
        "tools": [
            "python_repl", "create_chart", "parallel_search",
            "csv_reader", "web_scraper",
        ],
        "prompt": ANALYSIS_AGENT_PROMPT,
        "recursion_limit": 20,
        "timeout_seconds": 150.0,
    },
    "fact_checker": {
        "tools": [
            "web_search", "wikipedia", "wikidata",
            "google_scholar", "fetch_url",
        ],
        "prompt": FACT_CHECKER_PROMPT,
        "recursion_limit": 15,
        "timeout_seconds": 120.0,
    },
    "translation": {
        "tools": [
            "translate", "fetch_url", "pdf_reader",
        ],
        "prompt": TRANSLATION_AGENT_PROMPT,
        "recursion_limit": 10,
        "timeout_seconds": 60.0,
    },
}


class SpecialistAgent:
    """A specialist agent with a focused tool subset and system prompt.

    Each specialist wraps a LangGraph-based tool-calling agent (same
    ``create_agent`` used by the main ResearchAgent) but with only the
    tools relevant to its domain.
    """

    def __init__(
        self,
        name: str,
        tools: list,
        system_prompt: str,
        llm: ChatAnthropic,
        tool_health: dict,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.name = name
        self.recursion_limit = recursion_limit
        self.timeout_seconds = timeout_seconds

        self.tools, self.disabled_tools = get_available_tools(tools, tool_health)

        if not self.tools:
            # All tools disabled — agent can still answer from knowledge
            self.agent = None
        else:
            self.agent = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=system_prompt,
                debug=False,
            )

    async def run(self, task: str, callbacks: Optional[list] = None) -> str:
        """Run a task and return the result as a string.

        Bounds the inner agent call by ``self.timeout_seconds`` via
        ``asyncio.wait_for``. On timeout, returns a degraded string
        so the orchestrator's ``asyncio.gather`` keeps the phase alive
        instead of propagating the cancellation.

        Args:
            task: The specific task description for this specialist.
            callbacks: Optional LangChain callback handlers.

        Returns:
            The specialist's answer text, or a degraded message on
            timeout / error.
        """
        if self.agent is None:
            return f"[{self.name}] No tools available for this task."

        messages = [HumanMessage(content=task)]
        config = {"recursion_limit": self.recursion_limit}
        if callbacks:
            config["callbacks"] = callbacks

        try:
            result = await asyncio.wait_for(
                self.agent.ainvoke({"messages": messages}, config),
                timeout=self.timeout_seconds,
            )
            return self._extract_answer(result)
        except asyncio.TimeoutError:
            return (
                f"[{self.name}] Timed out after {self.timeout_seconds:.0f}s — "
                f"partial or no answer available. The orchestrator will "
                f"continue with other specialists."
            )
        except Exception as e:
            return f"[{self.name}] Error: {str(e)}"

    def _extract_answer(self, result: dict) -> str:
        """Extract text from the agent result (same pattern as ResearchAgent)."""
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                if isinstance(msg.content, list):
                    text_parts = [
                        block["text"] for block in msg.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if text_parts:
                        return "\n".join(text_parts)
        return "No answer was generated."


def build_specialists(
    all_tools: list,
    llm: ChatAnthropic,
    tool_health: dict,
) -> Dict[str, SpecialistAgent]:
    """Build all specialist agents from the master tool list.

    Each specialist gets only the tools defined in SPECIALIST_DEFINITIONS,
    filtered through the existing health-check system.

    Args:
        all_tools: The full list of LangChain Tool objects.
        llm: The ChatAnthropic LLM instance.
        tool_health: Output from check_tool_health().

    Returns:
        Dict mapping specialist names to SpecialistAgent instances.
    """
    tool_by_name = {t.name: t for t in all_tools}

    specialists = {}
    for name, defn in SPECIALIST_DEFINITIONS.items():
        specialist_tools = [
            tool_by_name[t] for t in defn["tools"]
            if t in tool_by_name
        ]
        specialists[name] = SpecialistAgent(
            name=name,
            tools=specialist_tools,
            system_prompt=defn["prompt"],
            llm=llm,
            tool_health=tool_health,
            recursion_limit=defn["recursion_limit"],
            timeout_seconds=defn["timeout_seconds"],
        )

    return specialists
