"""Multi-agent orchestrator.

Drives the supervisor + specialist pipeline through a single async event
generator that is the source of truth for all three public entry points:

  * ``run(query)``          — async, returns final answer
  * ``run_verbose(query)``  — async, prints CLI progress while streaming
  * ``stream(query)``       — sync generator for Streamlit

Pipeline:

  supervisor_plan  →  dispatch_phases  →  synthesize

Within each phase, independent specialists run concurrently via
``asyncio.gather()``.
"""

import asyncio
import queue
import sys
import threading
from typing import AsyncGenerator, Dict, Generator, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from src.multi_agent.supervisor import Supervisor, DelegationPlan
from src.multi_agent.specialists import SpecialistAgent, build_specialists


class MultiAgentOrchestrator:
    """Orchestrates multiple specialist agents via a supervisor.

    The orchestrator:
    1. Asks the supervisor to create a DelegationPlan.
    2. Dispatches specialists in phases — each phase runs its agents
       concurrently via ``asyncio.gather()``.
    3. Optionally runs the fact-checker on prior findings.
    4. Asks the supervisor to synthesize all outputs into a final answer.
    """

    def __init__(
        self,
        llm: ChatAnthropic,
        all_tools: list,
        tool_health: dict,
        callbacks: Optional[list] = None,
    ):
        self.llm = llm
        self.callbacks = callbacks or []
        self.supervisor = Supervisor(llm)
        self.specialists = build_specialists(all_tools, llm, tool_health)

    # ------------------------------------------------------------------
    # Single source of truth: async event generator
    # ------------------------------------------------------------------

    async def _astream_events(self, query: str) -> AsyncGenerator[dict, None]:
        """Yield typed events driving the whole multi-agent pipeline.

        Event types:

        * ``plan_created``       — delegation plan ready
        * ``phase_started``      — a new phase begins
        * ``specialist_started`` — a specialist is running
        * ``specialist_done``    — a specialist finished
        * ``phase_done``         — a phase completed
        * ``synthesis_token``    — one chunk of the final answer
        * ``done``               — final answer ready
        """
        # ---- Step 1: supervisor plan ------------------------------------
        plan = self.supervisor.create_delegation_plan(query)
        yield {"type": "plan_created", "plan": plan}

        # ---- Step 2: dispatch phases ------------------------------------
        all_results: Dict[str, str] = {}

        for phase_idx, phase in enumerate(plan.execution_phases):
            yield {
                "type": "phase_started",
                "phase_idx": phase_idx,
                "specialists": phase,
            }

            # Announce every specialist in this phase before launching,
            # so the UI can render them all at once.
            for name in phase:
                yield {
                    "type": "specialist_started",
                    "specialist": name,
                    "phase_idx": phase_idx,
                }

            # Build context string from prior phase results
            prior_context = ""
            if all_results:
                parts = [
                    f"[{name}]: {text[:500]}"
                    for name, text in all_results.items()
                ]
                prior_context = (
                    "\n\nContext from previous agents:\n" + "\n".join(parts)
                )

            async def _run_specialist(name: str) -> tuple:
                task = plan.specialist_tasks.get(name, query)

                # If this is the fact-checker, include prior findings
                if name == "fact_checker" and all_results:
                    findings_text = "\n\n".join(
                        f"[{n}]: {r}"
                        for n, r in all_results.items()
                        if n != "fact_checker"
                    )
                    task = (
                        f"Verify the key claims from these findings:\n\n"
                        f"{findings_text}\n\n"
                        f"Original question: {query}"
                    )
                elif prior_context and name != "fact_checker":
                    task = task + prior_context

                specialist = self.specialists.get(name)
                if specialist is None:
                    return name, f"[{name}] Unknown specialist."

                result = await specialist.run(task, callbacks=self.callbacks)
                return name, result

            # Run all specialists in this phase concurrently.
            phase_results = await asyncio.gather(
                *[_run_specialist(name) for name in phase]
            )

            for name, result in phase_results:
                all_results[name] = result
                yield {
                    "type": "specialist_done",
                    "specialist": name,
                    "phase_idx": phase_idx,
                    "result_preview": result[:300],
                }

            yield {"type": "phase_done", "phase_idx": phase_idx}

        # ---- Step 3: synthesize (token-streamed) ------------------------
        fact_check_report = all_results.pop("fact_checker", "")

        synthesis_input = (
            f"Synthesize these specialist findings into a comprehensive answer.\n\n"
            f"Original question: {query}\n\n"
            + "\n\n".join(
                f"--- {n.upper()} AGENT ---\n{r}"
                for n, r in all_results.items()
            )
            + (
                f"\n\n--- FACT-CHECK REPORT ---\n{fact_check_report}"
                if fact_check_report
                else ""
            )
        )

        final_answer = ""
        async for chunk in self.llm.astream(
            [HumanMessage(content=synthesis_input)]
        ):
            text = _extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": "synthesis_token", "token": text}

        yield {
            "type": "done",
            "answer": final_answer,
            "plan": plan,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> str:
        """Run the multi-agent pipeline and return the final answer."""
        final_answer = ""
        async for event in self._astream_events(query):
            if event.get("type") == "done":
                final_answer = event.get("answer", "")
        return final_answer

    async def run_verbose(self, query: str) -> str:
        """Run with verbose CLI output showing each phase and specialist."""
        print(f"\n{'=' * 60}")
        print("Multi-Agent Orchestration")
        print(f"{'=' * 60}")
        print(f"\nQuery: {query}\n")

        final_answer = ""
        synthesis_started = False

        async for event in self._astream_events(query):
            etype = event.get("type")

            if etype == "plan_created":
                plan = event["plan"]
                print("Supervisor is analyzing the query...")
                print(f"\nDelegation Plan ({len(plan.execution_phases)} phases):")
                print(f"  Rationale: {plan.rationale}")
                for i, phase in enumerate(plan.execution_phases):
                    parallel_note = " (parallel)" if len(phase) > 1 else ""
                    print(f"  Phase {i + 1}{parallel_note}: {', '.join(phase)}")
                if plan.needs_fact_check:
                    print("  Fact-checking: enabled")
                print()

            elif etype == "phase_started":
                phase = event["specialists"]
                parallel_note = " (parallel)" if len(phase) > 1 else ""
                print(f"{'─' * 40}")
                print(
                    f"Phase {event['phase_idx'] + 1}{parallel_note}: "
                    f"{', '.join(phase)}"
                )
                print(f"{'─' * 40}")

            elif etype == "specialist_started":
                print(f"  Starting {event['specialist']}...")

            elif etype == "specialist_done":
                print(f"  {event['specialist']} done.")

            elif etype == "synthesis_token":
                if not synthesis_started:
                    print(f"\n{'=' * 60}")
                    print("Synthesizing all findings...")
                    print(f"{'=' * 60}\n")
                    synthesis_started = True
                sys.stdout.write(event["token"])
                sys.stdout.flush()

            elif etype == "done":
                final_answer = event.get("answer", "")

        print()  # newline after streaming
        return final_answer

    def stream(self, query: str) -> Generator[dict, None, None]:
        """Sync generator that yields typed events for the Streamlit UI.

        Thin thread+queue bridge around :meth:`_astream_events` — the async
        generator runs on a dedicated worker thread so HTTP clients inside
        tools keep their connection pools across phases.
        """
        _SENTINEL = object()
        q: "queue.Queue" = queue.Queue()

        def _runner() -> None:
            async def _drive() -> None:
                try:
                    async for event in self._astream_events(query):
                        q.put(("event", event))
                except BaseException as exc:  # propagate to consumer
                    q.put(("error", exc))
                finally:
                    q.put(("done", _SENTINEL))

            asyncio.run(_drive())

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()

        while True:
            kind, payload = q.get()
            if kind == "event":
                yield payload
            elif kind == "error":
                worker.join()
                raise payload
            else:  # "done"
                break

        worker.join()


def _extract_chunk_text(chunk) -> str:
    """Extract text from an LLM chunk."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
