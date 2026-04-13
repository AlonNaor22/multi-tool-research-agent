"""Plan generation with complexity detection for plan-and-execute research mode."""

import json
import logging
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.constants import STATUS_PENDING, StepStatus
from src.utils import flatten_content

logger = logging.getLogger(__name__)

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
    """Multi-step research plan with dependency graph for parallel execution."""
    query: str
    steps: List[ResearchStep] = Field(default_factory=list)
    depends_on: Dict[int, List[int]] = Field(default_factory=dict)
    is_simple: bool = False


_PLANNER_SYSTEM = """\
You are a research planning assistant.
Given a complex research query, decompose it into 3–6 concrete research steps.

For each step specify:
- A clear description of what to research in this step
- Which tools would be most useful (choose from: web_search, wikipedia, arxiv_search,
  news_search, google_scholar, reddit_search, youtube_search, fetch_url, pdf_reader,
  python_repl, calculator, wolfram_alpha, weather, wikidata, translate)

RULES:
1. Use depends_on to declare which steps need another step's output.
2. Steps with no dependencies run in parallel automatically — don't make them sequential \
unless one truly needs the other's findings.
3. A final comparison/synthesis step should depend on all the research steps it draws from.

Respond with ONLY valid JSON (no markdown fences, no commentary):
{
  "steps": [
    {"step_number": 1, "description": "Research topic A", "expected_tools": ["web_search"]},
    {"step_number": 2, "description": "Research topic B", "expected_tools": ["web_search"]},
    {"step_number": 3, "description": "Compare A and B findings", "expected_tools": ["python_repl"]}
  ],
  "depends_on": {
    "1": [],
    "2": [],
    "3": [1, 2]
  }
}
"""


def _parse_depends_on(
    data: dict, valid_steps: set[int],
) -> Dict[int, List[int]]:
    """Extract and validate depends_on from LLM response JSON.

    Falls back to a sequential chain if the field is missing.
    """
    raw = data.get("depends_on")
    if not raw or not isinstance(raw, dict):
        # Fallback: sequential chain (step N depends on N-1)
        sorted_steps = sorted(valid_steps)
        deps: Dict[int, List[int]] = {sorted_steps[0]: []}
        for prev, cur in zip(sorted_steps, sorted_steps[1:]):
            deps[cur] = [prev]
        return deps

    deps = {}
    for key, val in raw.items():
        try:
            sn = int(key)
        except (ValueError, TypeError):
            continue
        if sn not in valid_steps:
            continue
        dep_list = [int(d) for d in val if int(d) in valid_steps and int(d) != sn]
        deps[sn] = dep_list

    # Ensure every step has an entry
    for sn in valid_steps:
        deps.setdefault(sn, [])

    return deps


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

        depends_on = _parse_depends_on(data, {s.step_number for s in steps})
        return ResearchPlan(query=query, steps=steps, depends_on=depends_on)

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
            depends_on={1: []},
        )
