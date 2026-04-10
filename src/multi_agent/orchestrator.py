"""Multi-agent orchestrator: supervisor planning, parallel specialist dispatch, synthesis."""

import asyncio
import queue
import sys
import threading
from typing import AsyncGenerator, Dict, Generator, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from src.constants import (
    SPECIALIST_FACT_CHECKER,
    EVENT_PLAN_CREATED, EVENT_PHASE_STARTED, EVENT_SPECIALIST_STARTED,
    EVENT_SPECIALIST_DONE, EVENT_PHASE_DONE, EVENT_SYNTHESIS_TOKEN,
    EVENT_DONE,
)
from src.utils import extract_chunk_text

from src.multi_agent.supervisor import Supervisor, DelegationPlan
from src.multi_agent.specialists import SpecialistAgent, build_specialists


class MultiAgentOrchestrator:
    """Phased parallel specialist orchestrator driven by a supervisor plan."""

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

    async def _astream_events(self, query: str) -> AsyncGenerator[dict, None]:
        """Yield typed events (plan, phase, specialist, synthesis, done) for the pipeline."""
        plan = await self.supervisor.acreate_delegation_plan(query)
        yield {"type": EVENT_PLAN_CREATED, "plan": plan}

        all_results: Dict[str, str] = {}

        for phase_idx, phase in enumerate(plan.execution_phases):
            yield {
                "type": EVENT_PHASE_STARTED,
                "phase_idx": phase_idx,
                "specialists": phase,
            }

            # Announce all specialists before launching so the UI renders them at once
            for name in phase:
                yield {
                    "type": EVENT_SPECIALIST_STARTED,
                    "specialist": name,
                    "phase_idx": phase_idx,
                }

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
                if name == SPECIALIST_FACT_CHECKER and all_results:
                    findings_text = "\n\n".join(
                        f"[{n}]: {r}"
                        for n, r in all_results.items()
                        if n != SPECIALIST_FACT_CHECKER
                    )
                    task = (
                        f"Verify the key claims from these findings:\n\n"
                        f"{findings_text}\n\n"
                        f"Original question: {query}"
                    )
                elif prior_context and name != SPECIALIST_FACT_CHECKER:
                    task = task + prior_context

                specialist = self.specialists.get(name)
                if specialist is None:
                    return name, f"[{name}] Unknown specialist."

                result = await specialist.run(task, callbacks=self.callbacks)
                return name, result

            phase_results = await asyncio.gather(
                *[_run_specialist(name) for name in phase]
            )

            for name, result in phase_results:
                all_results[name] = result
                yield {
                    "type": EVENT_SPECIALIST_DONE,
                    "specialist": name,
                    "phase_idx": phase_idx,
                    "result_preview": result[:300],
                }

            yield {"type": EVENT_PHASE_DONE, "phase_idx": phase_idx}

        fact_check_report = all_results.pop(SPECIALIST_FACT_CHECKER, "")

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
            text = extract_chunk_text(chunk)
            if text:
                final_answer += text
                yield {"type": EVENT_SYNTHESIS_TOKEN, "token": text}

        yield {
            "type": EVENT_DONE,
            "answer": final_answer,
            "plan": plan,
        }

    async def run(self, query: str) -> str:
        """Run the full pipeline and return the final synthesized answer."""
        final_answer = ""
        async for event in self._astream_events(query):
            if event.get("type") == EVENT_DONE:
                final_answer = event.get("answer", "")
        return final_answer

    async def run_verbose(self, query: str) -> str:
        """Run with verbose CLI output for each phase and specialist."""
        print(f"\n{'=' * 60}")
        print("Multi-Agent Orchestration")
        print(f"{'=' * 60}")
        print(f"\nQuery: {query}\n")

        final_answer = ""
        synthesis_started = False

        async for event in self._astream_events(query):
            etype = event.get("type")

            if etype == EVENT_PLAN_CREATED:
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

            elif etype == EVENT_PHASE_STARTED:
                phase = event["specialists"]
                parallel_note = " (parallel)" if len(phase) > 1 else ""
                print(f"{'─' * 40}")
                print(
                    f"Phase {event['phase_idx'] + 1}{parallel_note}: "
                    f"{', '.join(phase)}"
                )
                print(f"{'─' * 40}")

            elif etype == EVENT_SPECIALIST_STARTED:
                print(f"  Starting {event['specialist']}...")

            elif etype == EVENT_SPECIALIST_DONE:
                print(f"  {event['specialist']} done.")

            elif etype == EVENT_SYNTHESIS_TOKEN:
                if not synthesis_started:
                    print(f"\n{'=' * 60}")
                    print("Synthesizing all findings...")
                    print(f"{'=' * 60}\n")
                    synthesis_started = True
                sys.stdout.write(event["token"])
                sys.stdout.flush()

            elif etype == EVENT_DONE:
                final_answer = event.get("answer", "")

        print()  # newline after streaming
        return final_answer

    def stream(self, query: str) -> Generator[dict, None, None]:
        """Sync generator bridging async events via a thread+queue for Streamlit."""
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
