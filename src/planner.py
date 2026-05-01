"""Plan generation with complexity detection for plan-and-execute research mode."""

import logging
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.constants import STATUS_PENDING, StepStatus

logger = logging.getLogger(__name__)

# ─── Module overview ───────────────────────────────────────────────
# Generates a multi-step ResearchPlan from a user query via the LLM
# using Anthropic's native structured outputs (with_structured_output),
# with a dependency graph for parallel execution. Simple queries are
# detected early and skipped (is_simple=True).
# ───────────────────────────────────────────────────────────────────

_SIMPLE_STARTERS = (
    "what is ", "what are ", "who is ", "who are ", "when did ", "when is ",
    "where is ", "where are ", "how much ", "how many ", "convert ", "calculate ",
    "translate ", "weather in ", "weather for ", "define ", "what does ",
    "how do i ", "can you ",
)

_SIMPLE_MAX_WORDS = 8


# Takes (query). Returns True if the query is short or matches a simple-question pattern.
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


class _StepResponse(BaseModel):
    """LLM-facing schema for a single research step."""
    step_number: int = Field(description="1-indexed step number.")
    description: str = Field(description="What to research in this step.")
    expected_tools: List[str] = Field(
        default_factory=list,
        description="Tool names the agent should consider for this step.",
    )


class _PlanResponse(BaseModel):
    """LLM-facing schema for structured-output research planning."""
    steps: List[_StepResponse] = Field(
        default_factory=list,
        description="Ordered list of research steps (3–6 steps).",
    )
    depends_on: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Map of step_number → list of step_numbers it depends on. Empty list means no dependencies.",
    )


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
"""


# Takes (raw, valid_steps). Validates the LLM's depends_on map: drops unknown
# step references and self-loops, ensures every step has an entry.
# Falls back to a sequential chain if the map is empty.
def _parse_depends_on(
    raw: Dict[int, List[int]], valid_steps: set[int],
) -> Dict[int, List[int]]:
    """Validate depends_on map; falls back to a sequential chain if empty."""
    if not raw:
        sorted_steps = sorted(valid_steps)
        deps: Dict[int, List[int]] = {sorted_steps[0]: []}
        for prev, cur in zip(sorted_steps, sorted_steps[1:]):
            deps[cur] = [prev]
        return deps

    deps: Dict[int, List[int]] = {}
    for sn, dep_list in raw.items():
        if sn not in valid_steps:
            continue
        deps[sn] = [d for d in dep_list if d in valid_steps and d != sn]

    for sn in valid_steps:
        deps.setdefault(sn, [])

    return deps


# Takes (query, llm). Uses Anthropic's structured-output API to decompose a
# complex query into typed steps, or returns is_simple=True for trivial queries.
# Falls back to a single-step plan when the API call itself fails.
def generate_plan(query: str, llm: ChatAnthropic) -> ResearchPlan:
    """Return a ResearchPlan with steps, or is_simple=True for trivial queries."""
    if is_simple_query(query):
        return ResearchPlan(query=query, is_simple=True)

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=f"Create a research plan for: {query}"),
    ]

    try:
        planner = llm.with_structured_output(_PlanResponse)
        response = planner.invoke(messages)
        steps = [
            ResearchStep(
                step_number=s.step_number,
                description=s.description,
                expected_tools=s.expected_tools,
            )
            for s in response.steps
        ]
        if not steps:
            raise ValueError("Empty steps list")

        depends_on = _parse_depends_on(response.depends_on, {s.step_number for s in steps})
        return ResearchPlan(query=query, steps=steps, depends_on=depends_on)

    except Exception:
        logger.warning(
            "Falling back to single-step plan for query %r",
            query[:120],
            exc_info=True,
        )
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
