# TODO — Consolidated Project Backlog

Within each category, items are ordered by **importance** (highest first).

## Status legend
- [ ] not started
- [~] in progress
- [x] done

---

## Structured output & tool-schema enforcement (audit)

Background: `src/multi_agent/supervisor.py` was converted from `json.loads(llm.invoke(...))` to
`llm.with_structured_output(_PlanResponse)` so Anthropic's tool-use API enforces the Pydantic
schema. The same fragile pattern exists elsewhere — both in direct LLM calls and in tool
definitions that take a single `query: str` and parse it as JSON inside the function.

1. [x] **Convert `src/planner.py` to `with_structured_output`** —
   `src/planner.py`
   Replaced `llm.invoke()` + `json.loads()` with `llm.with_structured_output(_PlanResponse).invoke()`.
   Added LLM-facing `_StepResponse` / `_PlanResponse` schemas separate from the runtime
   `ResearchStep` / `ResearchPlan` (so the LLM never sees `status`/`findings`). Trimmed
   the JSON-formatting block from `_PLANNER_SYSTEM`. `_parse_depends_on` now takes the
   typed dict directly (Pydantic handled the `int(key)` parsing). Fallback path now logs
   a warning, matching the supervisor pattern.

2. [x] **Migrate tools that REQUIRE JSON input to `BaseTool` + `args_schema`** —
   `src/tools/parallel_tool.py`, `src/tools/datetime_tool.py`
   - `parallel_tool.py`: now `ParallelSearchTool(BaseTool)` with `ParallelSearchInput`
     (`searches: List[SearchSpec]`, `min_length=1`, `max_length=10`); `SearchSpec.type`
     is `Literal["web","wikipedia","news","arxiv"]`. `_arun` normalizes validated
     `SearchSpec` instances to dicts so the inner `parallel_search(searches)` stays
     Pydantic-agnostic.
   - `datetime_tool.py`: now `DatetimeTool(BaseTool)` with `DatetimeInput`; `operation`
     is `Literal["now","add","diff","convert","info","business_days"]`. Renamed JSON
     keys to Python-safe params (`from`→`date_from`, `to`→`date_to`,
     `datetime`→`convert_datetime`).
   - Tests rewritten: dropped JSON-parsing fixtures, added `TestParallelSearchSchema` /
     `TestDatetimeSchema` for Pydantic boundary validation. End-to-end `tool.ainvoke({...})`
     verified — schema rejects bad operations and out-of-range list sizes.

3. [ ] **Migrate string-or-JSON tools to `BaseTool` + `args_schema`** —
   `src/tools/search_tool.py`, `wikipedia_tool.py`, `news_tool.py`, `arxiv_tool.py`,
   `google_scholar_tool.py`, `reddit_tool.py`, `github_tool.py`, `csv_tool.py`, `scraper_tool.py`
   All nine use `parse_tool_input(query, defaults)` to extract optional args from a
   JSON-encoded query string. The LLM is told the tool takes a string but the docstring
   sometimes also tells it to pass JSON — schema is enforced nowhere. Convert each to
   named typed parameters (e.g. `web_search(query: str, max_results: int = 5, region: Optional[str] = None)`).
   After all nine are migrated, delete `parse_tool_input` from `src/utils.py:218`.

4. [ ] **Eliminate the `MATH_STRUCTURED:` text-channel** —
   `app.py:130-183`, `main.py:79`, `src/tools/math_formatter.py`
   `_auto_format_math_structured()` brace-counts to find a JSON blob inside LLM-streamed
   text, with `json.loads` as fallback. Same class as the recent `2772300 "chart path
   leaks as text"` bug — the agent sometimes echoes the raw `MATH_STRUCTURED:{…}` instead
   of routing it through `math_formatter`. Proper fix: either `calculator_tool` returns
   formatted output directly, or make `math_formatter` a guaranteed pipeline stage instead
   of an optional tool the LLM may forget to call.

5. [ ] **Structured fact-checker output** (optional, only if surfacing in UI) —
   `src/multi_agent/prompts.py:160-166`, `src/multi_agent/orchestrator.py:119-120`
   `FACT_CHECKER_PROMPT` asks for free-text `CONFIRMED:` / `CONTRADICTED:` / `UNVERIFIABLE:`
   lines. The fact-checker is a tool-using agent so its main call can't use
   `with_structured_output` directly, but a one-shot post-pass via
   `llm.with_structured_output(FactCheckReport)` (list of `{claim, verdict: Literal[...], source}`)
   would unlock per-verdict UI badges, telemetry on confidence ratios, and reliable
   downstream consumption by the synthesizer.

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

4. [x] **History management as agent middleware** — `src/agent.py`
   Original framing was stale (`SimpleMemory` was already deleted by the
   SqliteSaver migration). Reframed: SqliteSaver let history grow
   without bound. Added `_HistorySummarizerMiddleware` (subclass of
   `langchain.agents.middleware.AgentMiddleware`) with both
   `before_model` and `abefore_model` hooks. When the messages channel
   exceeds `HISTORY_TRIM_THRESHOLD_TOKENS` (8000), the middleware
   summarizes the older portion via the LLM into one marked
   `AIMessage` and keeps the active turn (last HumanMessage onward)
   verbatim plus as much earlier history as fits in
   `HISTORY_KEEP_RECENT_TOKENS` (4000). Tool-use/tool-result pairing
   is preserved across the drop/keep split. Telemetry: new
   `summarized_message_count` field on `ResearchAgentState`.

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
