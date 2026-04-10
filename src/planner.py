"""Plan generation with complexity detection for plan-and-execute research mode."""

import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.constants import STATUS_PENDING, StepStatus
from src.utils import flatten_content


_SIMPLE_STARTERS = (
    "what is ", "what are ", "who is ", "who are ", "when did ", "when is ",
    "where is ", "where are ", "how much ", "how many ", "convert ", "calculate ",
    "translate ", "weather in ", "weather for ", "define ", "what does ",
    "how do i ", "can you ",
)

_SIMPLE_MAX_WORDS = 8


def is_simple_query(query: str) -> bool:
    """Return True if query is short or starts with a simple-question pattern."""
    q = query.strip().lower()
    if len(q.split()) <= _SIMPLE_MAX_WORDS:
        return True
    for starter in _SIMPLE_STARTERS:
        if q.startswith(starter):
            return True
    return False


class ResearchStep(BaseModel):
    """Single numbered step with description, expected tools, status, and findings."""
    step_number: int
    description: str
    expected_tools: List[str] = Field(default_factory=list)
    status: StepStatus = STATUS_PENDING
    findings: str = ""


class ResearchPlan(BaseModel):
    """Multi-step research plan; is_simple=True bypasses planning."""
    query: str
    steps: List[ResearchStep] = Field(default_factory=list)
    is_simple: bool = False


_PLANNER_SYSTEM = """\
You are a research planning assistant.
Given a complex research query, decompose it into 3–6 concrete, sequential research steps.

For each step specify:
- A clear description of what to research in this step
- Which tools would be most useful (choose from: web_search, wikipedia, arxiv_search,
  news_search, google_scholar, reddit_search, youtube_search, fetch_url, pdf_reader,
  python_repl, calculator, wolfram_alpha, weather, wikidata, translate)

Respond with ONLY valid JSON (no markdown fences, no commentary):
{
  "steps": [
    {
      "step_number": 1,
      "description": "...",
      "expected_tools": ["tool1", "tool2"]
    }
  ]
}
"""


def generate_plan(query: str, llm: ChatAnthropic) -> ResearchPlan:
    """Return a ResearchPlan with steps, or is_simple=True for trivial queries."""
    if is_simple_query(query):
        return ResearchPlan(query=query, is_simple=True)

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=f"Create a research plan for: {query}"),
    ]

    try:
        response = llm.invoke(messages)
        content = flatten_content(response.content)
        data = json.loads(content)
        steps = [
            ResearchStep(
                step_number=s["step_number"],
                description=s["description"],
                expected_tools=s.get("expected_tools", []),
            )
            for s in data.get("steps", [])
        ]
        if not steps:
            raise ValueError("Empty steps list")
        return ResearchPlan(query=query, steps=steps)

    except Exception:
        return ResearchPlan(
            query=query,
            steps=[
                ResearchStep(
                    step_number=1,
                    description=query,
                    expected_tools=["web_search", "wikipedia"],
                )
            ],
        )
