"""Tests for the multi-agent orchestration system.

Tests cover:
- Supervisor delegation plan generation and parsing
- DelegationPlan model validation
- Specialist agent definitions and tool mapping
- Orchestrator graph structure
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# DelegationPlan model tests
# ---------------------------------------------------------------------------

class TestDelegationPlan:
    """Tests for the DelegationPlan Pydantic model."""

    def test_basic_plan(self):
        from src.multi_agent.supervisor import DelegationPlan

        plan = DelegationPlan(
            query="test query",
            specialists=["research"],
            specialist_tasks={"research": "do research"},
            depends_on={"research": []},
        )
        assert plan.query == "test query"
        assert plan.specialists == ["research"]
        assert plan.depends_on == {"research": []}
        assert not plan.needs_fact_check

    def test_multi_phase_plan(self):
        from src.multi_agent.supervisor import DelegationPlan

        plan = DelegationPlan(
            query="complex query",
            specialists=["research", "math", "analysis"],
            specialist_tasks={
                "research": "gather data",
                "math": "calculate",
                "analysis": "analyze",
            },
            depends_on={"research": [], "math": [], "analysis": ["research", "math"]},
            needs_fact_check=True,
            rationale="Multi-domain query",
        )
        assert plan.specialists == ["research", "math", "analysis"]
        assert plan.depends_on == {"research": [], "math": [], "analysis": ["research", "math"]}
        assert plan.needs_fact_check

    def test_empty_plan_defaults(self):
        from src.multi_agent.supervisor import DelegationPlan

        plan = DelegationPlan(query="q")
        assert plan.specialists == []
        assert plan.depends_on == {}
        assert plan.specialist_tasks == {}
        assert plan.needs_fact_check is False
        assert plan.rationale == ""

    def test_serialization_roundtrip(self):
        from src.multi_agent.supervisor import DelegationPlan

        plan = DelegationPlan(
            query="test",
            specialists=["research", "math"],
            specialist_tasks={"research": "r", "math": "m"},
            depends_on={"research": [], "math": []},
            needs_fact_check=True,
        )
        data = plan.model_dump()
        restored = DelegationPlan(**data)
        assert restored.query == plan.query
        assert restored.specialists == plan.specialists
        assert restored.depends_on == plan.depends_on
        assert restored.needs_fact_check == plan.needs_fact_check


# ---------------------------------------------------------------------------
# Specialist definitions tests
# ---------------------------------------------------------------------------

class TestSpecialistDefinitions:
    """Tests for the specialist definitions and tool mappings."""

    def test_all_specialists_defined(self):
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        expected = {"research", "math", "analysis", "fact_checker", "translation"}
        assert set(SPECIALIST_DEFINITIONS.keys()) == expected

    def test_each_specialist_has_tools_and_prompt(self):
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        for name, defn in SPECIALIST_DEFINITIONS.items():
            assert "tools" in defn, f"{name} missing 'tools'"
            assert "prompt" in defn, f"{name} missing 'prompt'"
            assert isinstance(defn["tools"], list), f"{name} tools not a list"
            assert len(defn["tools"]) > 0, f"{name} has no tools"
            assert isinstance(defn["prompt"], str), f"{name} prompt not a string"
            assert len(defn["prompt"]) > 50, f"{name} prompt too short"

    def test_research_agent_has_broad_tools(self):
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        research_tools = SPECIALIST_DEFINITIONS["research"]["tools"]
        assert "web_search" in research_tools
        assert "wikipedia" in research_tools
        assert "arxiv_search" in research_tools

    def test_math_agent_has_computation_tools(self):
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        math_tools = SPECIALIST_DEFINITIONS["math"]["tools"]
        assert "calculator" in math_tools
        assert "equation_solver" in math_tools
        assert "python_repl" in math_tools

    def test_fact_checker_has_verification_tools(self):
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        fc_tools = SPECIALIST_DEFINITIONS["fact_checker"]["tools"]
        assert "web_search" in fc_tools
        assert "wikipedia" in fc_tools
        assert "wikidata" in fc_tools

    def test_fact_checker_overlaps_with_research(self):
        """Fact-checker should share tools with research for independent verification."""
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        research = set(SPECIALIST_DEFINITIONS["research"]["tools"])
        fc = set(SPECIALIST_DEFINITIONS["fact_checker"]["tools"])
        overlap = research & fc
        assert len(overlap) >= 3, "Fact-checker needs overlapping tools for cross-verification"


# ---------------------------------------------------------------------------
# Supervisor logic tests
# ---------------------------------------------------------------------------

class TestSupervisor:
    """Tests for the Supervisor class."""

    def test_delegation_plan_parsing(self):
        """Test that the supervisor correctly parses LLM JSON output."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "specialists": ["research", "math", "analysis"],
            "specialist_tasks": {
                "research": "find data",
                "math": "calculate",
                "analysis": "chart it",
            },
            "depends_on": {"research": [], "math": [], "analysis": ["research", "math"]},
            "needs_fact_check": False,
            "rationale": "multi-domain query",
        })
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("test query")

        assert plan.specialists == ["research", "math", "analysis"]
        assert plan.depends_on == {"research": [], "math": [], "analysis": ["research", "math"]}
        assert not plan.needs_fact_check

    def test_fact_check_appended_as_final_phase(self):
        """When needs_fact_check=True, fact_checker should be the last phase."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "specialists": ["research"],
            "specialist_tasks": {"research": "find info"},
            "depends_on": {"research": []},
            "needs_fact_check": True,
            "rationale": "verification needed",
        })
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("verify this claim")

        assert plan.needs_fact_check
        assert "fact_checker" in plan.specialists
        assert plan.depends_on["fact_checker"] == ["research"]
        assert "fact_checker" in plan.specialist_tasks

    def test_fallback_on_invalid_json(self):
        """Supervisor should fall back to research-only on parse failure."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "this is not valid json!!"
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("some query")

        assert plan.specialists == ["research"]
        assert plan.specialist_tasks == {"research": "some query"}

    def test_invalid_specialist_names_filtered(self):
        """Unknown specialist names in the LLM output should be filtered out."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "specialists": ["research", "nonexistent_agent", "math"],
            "specialist_tasks": {
                "research": "r",
                "nonexistent_agent": "n",
                "math": "m",
            },
            "depends_on": {"research": [], "nonexistent_agent": [], "math": []},
            "needs_fact_check": False,
        })
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("test")

        # nonexistent_agent should be filtered out
        assert "nonexistent_agent" not in plan.specialists
        assert "research" in plan.specialists
        assert "math" in plan.specialists

    def test_empty_phases_fallback(self):
        """If all specialist names are invalid, fall back to research."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "specialists": ["bogus1", "bogus2"],
            "specialist_tasks": {},
            "depends_on": {"bogus1": [], "bogus2": []},
            "needs_fact_check": False,
        })
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("test")

        # Should fall back
        assert plan.specialists == ["research"]

    def test_list_content_handling(self):
        """Test that list-format content blocks are handled correctly."""
        from src.multi_agent.supervisor import Supervisor

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": json.dumps({
                "specialists": ["research"],
                "specialist_tasks": {"research": "task"},
                "depends_on": {"research": []},
                "needs_fact_check": False,
            })}
        ]
        mock_llm.invoke.return_value = mock_response

        supervisor = Supervisor(mock_llm)
        plan = supervisor.create_delegation_plan("test")

        assert plan.specialists == ["research"]


# ---------------------------------------------------------------------------
# Prompts tests
# ---------------------------------------------------------------------------

class TestPrompts:
    """Tests for prompt module constants."""

    def test_all_prompts_exist(self):
        from src.multi_agent.prompts import (
            SUPERVISOR_PLAN_PROMPT,
            SUPERVISOR_SYNTHESIZE_PROMPT,
            RESEARCH_AGENT_PROMPT,
            MATH_AGENT_PROMPT,
            ANALYSIS_AGENT_PROMPT,
            FACT_CHECKER_PROMPT,
            TRANSLATION_AGENT_PROMPT,
        )
        prompts = [
            SUPERVISOR_PLAN_PROMPT, SUPERVISOR_SYNTHESIZE_PROMPT,
            RESEARCH_AGENT_PROMPT, MATH_AGENT_PROMPT,
            ANALYSIS_AGENT_PROMPT, FACT_CHECKER_PROMPT,
            TRANSLATION_AGENT_PROMPT,
        ]
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 100, "Prompt too short to be useful"

    def test_supervisor_plan_prompt_mentions_json(self):
        from src.multi_agent.prompts import SUPERVISOR_PLAN_PROMPT
        assert "JSON" in SUPERVISOR_PLAN_PROMPT
        assert "specialists" in SUPERVISOR_PLAN_PROMPT
        assert "depends_on" in SUPERVISOR_PLAN_PROMPT

    def test_supervisor_plan_lists_all_specialists(self):
        from src.multi_agent.prompts import SUPERVISOR_PLAN_PROMPT
        for name in ["research", "math", "analysis", "fact_checker", "translation"]:
            assert name in SUPERVISOR_PLAN_PROMPT, f"{name} not in supervisor prompt"

    def test_fact_checker_prompt_mentions_verification(self):
        from src.multi_agent.prompts import FACT_CHECKER_PROMPT
        assert "verify" in FACT_CHECKER_PROMPT.lower() or "verif" in FACT_CHECKER_PROMPT.lower()
        assert "CONFIRMED" in FACT_CHECKER_PROMPT
        assert "CONTRADICTED" in FACT_CHECKER_PROMPT


# ---------------------------------------------------------------------------
# SpecialistAgent tests
# ---------------------------------------------------------------------------

class TestSpecialistAgent:
    """Tests for the SpecialistAgent class."""

    def test_no_tools_returns_message(self):
        """Agent with no available tools should return a message, not crash."""
        from src.multi_agent.specialists import SpecialistAgent

        agent = SpecialistAgent(
            name="test",
            tools=[],
            system_prompt="test prompt",
            llm=MagicMock(),
            tool_health={},
        )
        assert agent.agent is None

    def test_definitions_include_limits(self):
        """Every specialist definition should declare its resource limits."""
        from src.multi_agent.specialists import SPECIALIST_DEFINITIONS

        for name, defn in SPECIALIST_DEFINITIONS.items():
            assert "recursion_limit" in defn, f"{name} missing recursion_limit"
            assert "timeout_seconds" in defn, f"{name} missing timeout_seconds"
            assert defn["recursion_limit"] >= 5, (
                f"{name} recursion_limit too small"
            )
            assert defn["timeout_seconds"] >= 30, (
                f"{name} timeout_seconds too small"
            )

    async def test_run_times_out_gracefully(self):
        """A hung inner agent should trip the wait_for and return a
        degraded string instead of raising — so asyncio.gather keeps
        the phase alive."""
        import asyncio
        from src.multi_agent.specialists import SpecialistAgent

        fake_tool = MagicMock()
        fake_tool.name = "fake"

        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(5)
            return {"messages": []}

        with patch(
            "src.multi_agent.specialists.create_agent"
        ) as mock_create:
            inner_agent = MagicMock()
            inner_agent.ainvoke = slow_ainvoke
            mock_create.return_value = inner_agent

            agent = SpecialistAgent(
                name="slowpoke",
                tools=[fake_tool],
                system_prompt="sys",
                llm=MagicMock(),
                tool_health={"fake": {"available": True, "reason": None}},
                recursion_limit=5,
                timeout_seconds=0.2,
            )

            result = await agent.run("task")
            assert "Timed out" in result
            assert "slowpoke" in result

    async def test_run_error_returns_degraded_string(self):
        """Non-timeout exceptions should also be caught and returned,
        not propagated to the orchestrator's asyncio.gather."""
        from src.multi_agent.specialists import SpecialistAgent

        fake_tool = MagicMock()
        fake_tool.name = "fake"

        async def boom(*args, **kwargs):
            raise RuntimeError("tool blew up")

        with patch(
            "src.multi_agent.specialists.create_agent"
        ) as mock_create:
            inner_agent = MagicMock()
            inner_agent.ainvoke = boom
            mock_create.return_value = inner_agent

            agent = SpecialistAgent(
                name="crash",
                tools=[fake_tool],
                system_prompt="sys",
                llm=MagicMock(),
                tool_health={"fake": {"available": True, "reason": None}},
                timeout_seconds=5.0,
            )

            result = await agent.run("task")
            assert "Error" in result
            assert "tool blew up" in result

    def test_build_specialists_creates_all(self):
        """build_specialists should create all 5 specialist agents."""
        from src.multi_agent.specialists import build_specialists, SPECIALIST_DEFINITIONS

        # Create mock tools matching the names specialists expect
        all_tool_names = set()
        for defn in SPECIALIST_DEFINITIONS.values():
            all_tool_names.update(defn["tools"])

        mock_tools = []
        for name in all_tool_names:
            tool = MagicMock()
            tool.name = name
            mock_tools.append(tool)

        mock_llm = MagicMock()

        with patch("src.multi_agent.specialists.create_agent", return_value=MagicMock()):
            specialists = build_specialists(mock_tools, mock_llm, {})

        assert set(specialists.keys()) == set(SPECIALIST_DEFINITIONS.keys())
        for name, agent in specialists.items():
            assert agent.name == name


# ---------------------------------------------------------------------------
# Module exports test
# ---------------------------------------------------------------------------

class TestExports:
    """Test that the multi_agent package exports are correct."""

    def test_package_exports(self):
        from src.multi_agent import (
            MultiAgentOrchestrator,
            Supervisor,
            DelegationPlan,
            SpecialistAgent,
            build_specialists,
        )
        # Just verify they're importable
        assert MultiAgentOrchestrator is not None
        assert Supervisor is not None
        assert DelegationPlan is not None
        assert SpecialistAgent is not None
        assert build_specialists is not None
