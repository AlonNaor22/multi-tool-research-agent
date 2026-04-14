# TODO — Consolidated Project Backlog

Within each category, items are ordered by **importance** (highest first).

## Status legend
- [ ] not started
- [~] in progress
- [x] done

---

## LangGraph architecture & orchestration

1. [x] **Plan-execute runs steps strictly sequentially** (audit #4) —
   `src/agent.py`
   Added `depends_on` to `ResearchPlan`; wave-based parallel execution
   with `asyncio.gather` and async-to-sync bridge for Streamlit streaming.
   (commit 3cac7e3)

---

## Agent state & graph improvements (from state audit)

Do in this order — each builds on the previous:

1. [x] **Explicit state schema** — `src/agent.py`
   Defined `ResearchAgentState(AgentState)` and passed as `state_schema`
   to `create_agent()`. Specialists use base `AgentState` explicitly.

2. [x] **Unified persistence with SqliteSaver** — `src/agent.py`, `src/session_manager.py`
   Replaced SimpleMemory + JSON session files with SqliteSaver
   checkpointing to `sessions/checkpoints.db`. Auto-persists full
   graph state after every node. Sessions listed/loaded/deleted from
   SQLite. No manual "Save" button needed.

3. [x] **Inject dependency context as messages** — `src/agent.py`
   `_run_step` now replays each declared dependency as a
   `(HumanMessage, AIMessage)` pair before the current step's task,
   so prior findings arrive as real prior assistant turns. Also
   replaced sync `SqliteSaver` with `AsyncSqliteSaver` (bootstrapped
   on a persistent daemon loop) to unblock async `ainvoke`/`astream`.

4. [ ] **Memory as a LangGraph node** — `src/agent.py`
   `SimpleMemory` is manually prepended in Python before each invoke.
   Formalize as a graph node for LLM-aware summarization/pruning of
   long conversation history. (depends on #1 and #2)

---

## Code cleanup & refactoring

*All items completed.*

---

## Capabilities — new features

1. [ ] **RAG Pipeline & Semantic Memory**
   - Vector store (ChromaDB/FAISS) for embedding-based retrieval
   - Store past research results as embeddings
   - Query semantic memory before external API calls
   - Document ingestion (PDFs/URLs → chunk → embed → store)

2. [ ] **Human-in-the-Loop Interaction**
   - Agent asks clarifying questions before proceeding
   - Intermediate results for user approval
   - Confidence-based escalation

---

## Deployment & production readiness

1. [ ] FastAPI wrapper exposing the agent as a REST API
2. [ ] Dockerfile + docker-compose for containerized deployment
3. [ ] CI/CD pipeline (GitHub Actions) for tests, linting, build validation
4. [ ] Environment-based configuration (dev / staging / prod)
5. [ ] API authentication and rate limiting at the endpoint level

---

## Done (for context)

### Architecture audit
- [x] **Delete dead `_build_plan_execute_graph()`** — `src/agent.py`
- [x] **Delete dead `Supervisor.synthesize()`** — `src/multi_agent/supervisor.py`
- [x] **Wire `SUPERVISOR_SYNTHESIZE_PROMPT` into orchestrator synthesis** — `src/multi_agent/orchestrator.py`
- [x] **Add metrics save to multi-agent mode** — `src/agent.py`
- [x] **Add callbacks to plan-execute methods** — `src/agent.py`
- [x] **Centralize mode selection into `route_query()`** — `src/agent.py`
- [x] **Add error handling to plan-execute** — `src/agent.py`
- [x] **Unify orchestrator around single async event generator** (audit #1, #7, #12)
- [x] **Fold no-op `replan` node into `execute_step`** (audit #3, #13)
- [x] **Async `create_delegation_plan`** (audit #8)
- [x] **Per-specialist `recursion_limit` + `timeout_seconds`** (audit #9)
- [x] **Consolidate content-block flattener** (audit #11)
- [x] **Log supervisor fallback exceptions** (audit #14)

### LangGraph architecture & orchestration
- [x] **depends_on + fan-out for orchestrator phases** — wave-based parallel dispatch with `asyncio.gather`, context only from declared deps (commit 2d43a74)
- [x] **Pydantic `model_dump` round-trip** — code already uses `model_copy()` efficiently; no round-trip issue exists
- [x] **Plan-execute parallel steps** — added `depends_on` to `ResearchPlan`, wave-based execution with async-to-sync bridge (commit 3cac7e3)

### Context & token efficiency
- [x] **`prior_context` truncation + quadratic growth** — resolved by `depends_on` approach; no 500-char truncation exists, context injected only from declared dependencies

### Code cleanup
- [x] **Research agent prompt vague about tool names** — prompt now lists specific tools (web_search, arxiv_search, google_scholar, etc.)
- [x] **SPECIALIST_DEFINITIONS stringly-typed dict** — converted to `SpecialistConfig` frozen dataclass
- [x] **Timeout sentinel vs plain degraded string** — `SpecialistResult` dataclass with `timed_out`/`error` flags; synthesis skips timed-out specialists
- [x] **Sync-tool cancellation on timeout** — logged warning on timeout; Python threads can't be forcibly killed, documented as known limitation

### Codebase improvements
- [x] Migrate tools to `@tool` decorator + `BaseTool` subclass
- [x] Extract hardcoded strings to constants + localization file
- [x] Consolidate duplicated boilerplate across 21 tool files
- [x] Apply `@safe_tool_call` + `require_input` across 21 tools
- [x] Trim all docstrings to 2 lines
- [x] Optimize test suite: 84s → 12s (parallel, fast retries, dedup)

### IMPROVEMENTS.md (items 1–13, all done)
- [x] #1 Tests
- [x] #2 Migrate to native tool calling
- [x] #3 Evaluation suite
- [x] #4 Streaming output
- [x] #5 Better error recovery
- [x] #6 Web UI
- [x] #7 Observability
- [x] #8 Async / true parallelism
- [x] #9 Config cleanup
- [x] #10 Rate limiting
- [x] #11 Session schema versioning
- [x] #12 Plan-and-execute loop
- [x] #13 Multi-agent orchestration
