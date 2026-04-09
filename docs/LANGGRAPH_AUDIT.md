# LangGraph Nodes Audit — Optimization Tracker

Audit of LangGraph node implementations across `src/agent.py` (plan-execute graph)
and `src/multi_agent/orchestrator.py` (multi-agent orchestration graph).

Issues are ordered by implementation priority: local wins first, deeper refactors last.

## Status legend
- [ ] not started
- [~] in progress
- [x] done

---

## High-impact

### 1. [x] `MultiAgentOrchestrator.stream()` duplicates the whole graph
**Files:** `src/multi_agent/orchestrator.py`

Three copies of the pipeline previously existed: `_build_graph()`, `run_verbose()`,
and `stream()`. `stream()` reimplemented the pipeline in a sync generator entirely
outside the compiled graph, with dead code around `loop.run_in_executor`, and each
phase called `_asyncio.run()` — killing HTTP connection reuse.

**Done:** Introduced `async def _astream_events(query)` as the single source of
truth. `run()`, `run_verbose()`, and sync `stream()` now all consume it. Sync
`stream()` is a thin thread+queue bridge running `_astream_events` on a dedicated
worker thread so tool HTTP pools survive across phases. The `_build_graph()` /
`MultiAgentState` / `agent_activity` scaffolding was removed — it was dead weight
since the compiled graph was only used by `run()` and nothing downstream consumed
the graph state. File shrank from 453 → ~290 lines. All 23 multi-agent tests pass.

### 2. [ ] Phases serial even when independent; prior context injected blindly
**Files:** `src/multi_agent/orchestrator.py:92-101,118-119`

`dispatch_phases_node` appends truncated prior context to every specialist in
later phases regardless of whether it's needed. Forces serial execution even when
the supervisor could have flattened to one phase.

**Fix:** Add `depends_on: Dict[str, List[str]]` to `DelegationPlan`. Inject context
only from declared dependencies. Consider LangGraph's `Send` API to express fan-out
natively instead of hand-rolled `asyncio.gather` inside a single node.

### 3. [ ] Plan-execute graph has a no-op `replan` node
**Files:** `src/agent.py:662-665`

`replan_node` only increments a counter — can be folded into `execute_step_node`'s
return dict. Removes a graph hop per step. The docstring promises replanning but
the node does nothing.

**Fix:** Either implement real replanning (pass findings back to planner when
confidence drops) or delete the node and simplify edges.

### 4. [ ] Plan-execute runs steps strictly sequentially
**Files:** `src/agent.py:628-659`

No parallelism for independent research steps.

**Fix:** Add `depends_on` to `PlanStep`; use `asyncio.gather` or LangGraph `Send`
to batch independent steps.

---

## Medium-impact

### 5. [ ] Pydantic model_dump round-trip on every node call
**Files:** `src/agent.py:629,651,655`; `src/multi_agent/orchestrator.py:80`

`ResearchPlan(**state["plan_data"])` → mutate → `plan.model_dump()` every call.
O(plan_size) serialization for no reason — LangGraph only needs the delta.

**Fix:** Store plan data once; keep mutable step state as separate keys with
commutative reducers (`operator.or_` on dicts keyed by step index).

### 6. [ ] `prior_context` truncates at 500 chars arbitrarily
**Files:** `src/multi_agent/orchestrator.py:95,242,334`

500 chars ≈ 100 tokens — too short to carry findings, too long to be free.
Every phase pays for every prior specialist regardless of relevance.

**Fix:** Either pass full prior outputs (trust the LLM) or a supervisor-generated
summary per phase boundary.

### 7. [ ] `run_verbose` runs synthesis twice
**Files:** `src/multi_agent/orchestrator.py:280-298`

Streams synthesis via `self.llm.astream` directly, bypassing `synthesize_node`.
`run_verbose` isn't even using the graph it just built. Resolved by fix #1.

### 8. [ ] `create_delegation_plan` is sync inside async nodes
**Files:** `src/multi_agent/orchestrator.py:65,221,317`; `src/multi_agent/supervisor.py:46-59`

Sync `self.llm.invoke()` blocks the event loop during the planner call.

**Fix:** Make `create_delegation_plan` async; use `await self.llm.ainvoke(...)`.

### 9. [ ] No per-specialist recursion_limit or timeouts
**Files:** `src/multi_agent/specialists.py:115`

Hardcoded `recursion_limit: 20`. One stuck specialist blocks its whole phase via
`asyncio.gather` with no cancellation.

**Fix:** Per-specialist config for `recursion_limit` and `timeout`. Wrap
`specialist.run()` in `asyncio.wait_for` with a degraded-result fallback.

### 10. [ ] `prior_context` grows quadratically
**Files:** `src/multi_agent/orchestrator.py:92-101`

Fine for current scale, but worth noting. Same mitigation as #6.

---

## Low-impact / cleanup

### 11. [ ] `_extract_chunk_text` / `_extract_answer` duplicated three times
**Files:** `src/multi_agent/orchestrator.py:442`; `src/multi_agent/specialists.py:127`;
`src/agent.py` (inline)

**Fix:** Consolidate into `src/utils.py`.

### 12. [ ] `agent_activity` reducer is `operator.add` on list-of-dicts
**Files:** `src/multi_agent/orchestrator.py:33`

Copies whole list on every merge. Low priority — resolved or obsoleted by fix #1.

### 13. [ ] Plan-execute edges convoluted once `replan` is removed
**Files:** `src/agent.py:701-718`

Folds into fix #3.

### 14. [ ] `DelegationPlan` fallback silently drops exceptions
**Files:** `src/multi_agent/supervisor.py:112-120`

Bare `except` hides parser failures.

**Fix:** Log exception with context before falling back.

---

## Recommended ordering
1. Fix #1 (delete `stream()` duplication) — local, biggest cleanup
2. Fix #3 (fold `replan` into `execute_step`)
3. Fix #8 (async `create_delegation_plan`)
4. Fix #9 (per-specialist timeouts)
5. Fixes #2 + #4 (`depends_on` + LangGraph `Send` fan-out) — bigger refactor
6. Cleanups: #11, #14, #6
