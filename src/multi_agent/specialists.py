"""Specialist agents wrapping focused tool subsets for multi-agent dispatch."""

import asyncio
from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from src.constants import (
    SPECIALIST_RESEARCH, SPECIALIST_MATH, SPECIALIST_ANALYSIS,
    SPECIALIST_FACT_CHECKER, SPECIALIST_TRANSLATION,
)
from src.tool_health import get_available_tools
from src.utils import extract_ai_answer
from src.multi_agent.prompts import (
    RESEARCH_AGENT_PROMPT,
    MATH_AGENT_PROMPT,
    ANALYSIS_AGENT_PROMPT,
    FACT_CHECKER_PROMPT,
    TRANSLATION_AGENT_PROMPT,
)


DEFAULT_RECURSION_LIMIT = 20
DEFAULT_TIMEOUT_SECONDS = 120.0


SPECIALIST_DEFINITIONS = {
    SPECIALIST_RESEARCH: {
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
    SPECIALIST_MATH: {
        "tools": [
            "calculator", "unit_converter", "equation_solver",
            "currency_converter", "wolfram_alpha", "python_repl",
            "datetime_calculator", "math_formatter", "create_chart",
        ],
        "prompt": MATH_AGENT_PROMPT,
        "recursion_limit": 15,
        "timeout_seconds": 90.0,
    },
    SPECIALIST_ANALYSIS: {
        "tools": [
            "python_repl", "create_chart", "parallel_search",
            "csv_reader", "web_scraper",
        ],
        "prompt": ANALYSIS_AGENT_PROMPT,
        "recursion_limit": 20,
        "timeout_seconds": 150.0,
    },
    SPECIALIST_FACT_CHECKER: {
        "tools": [
            "web_search", "wikipedia", "wikidata",
            "google_scholar", "fetch_url",
        ],
        "prompt": FACT_CHECKER_PROMPT,
        "recursion_limit": 15,
        "timeout_seconds": 120.0,
    },
    SPECIALIST_TRANSLATION: {
        "tools": [
            "translate", "fetch_url", "pdf_reader",
        ],
        "prompt": TRANSLATION_AGENT_PROMPT,
        "recursion_limit": 10,
        "timeout_seconds": 60.0,
    },
}


class SpecialistAgent:
    """Domain-focused agent with a tool subset, recursion limit, and timeout."""

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
            self.agent = None
        else:
            self.agent = create_agent(
                model=llm,
                tools=self.tools,
                system_prompt=system_prompt,
                debug=False,
            )

    async def run(self, task: str, callbacks: Optional[list] = None) -> str:
        """Execute task with timeout; return answer string or degraded message."""
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
            return extract_ai_answer(result)
        except asyncio.TimeoutError:
            return (
                f"[{self.name}] Timed out after {self.timeout_seconds:.0f}s — "
                f"partial or no answer available. The orchestrator will "
                f"continue with other specialists."
            )
        except Exception as e:
            return f"[{self.name}] Error: {str(e)}"

def build_specialists(
    all_tools: list,
    llm: ChatAnthropic,
    tool_health: dict,
) -> Dict[str, SpecialistAgent]:
    """Build all specialists from SPECIALIST_DEFINITIONS with health-filtered tools."""
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
