# Planned Improvements

Prioritized list of improvements to make this project stand out as a portfolio piece for agent engineering roles.

## High Impact

### 1. ~~Tests~~ (DONE)
- [x] 283 tests across all tools, memory, sessions, callbacks, observability, and utilities

### 2. ~~Migrate to Native Tool Calling~~ (DONE)
- [x] Replaced `create_react_agent` (string-based ReAct parsing) with `create_agent` (LangGraph-based)
- [x] Uses Claude's native structured tool-use API instead of fragile text parsing
- [x] Messages-based conversation history instead of string concatenation
- [x] Updated requirements to langchain 1.x + langgraph

### 3. ~~Evaluation Suite~~ (DONE)
- [x] 15 curated test cases across all tool categories (math, search, weather, code, multi-step, ambiguous)
- [x] Two independent scores: tool selection accuracy + answer quality
- [x] CLI runner with `--case` and `--category` filtering, timestamped JSON reports
- [x] First run: 100% tool selection, 93% answer quality

### 4. ~~Streaming Output~~ (DONE)
- [x] `StreamingCallbackHandler` streams thinking/tool-use/answer tokens in real-time
- [x] `stream_query()` method uses LangGraph `stream()` instead of blocking `invoke()`
- [x] Original `query()` preserved for backward compatibility (tests/evals)

### 5. ~~Better Error Recovery~~ (DONE)
- [x] `check_tool_health()` probes API keys and libraries at startup
- [x] `get_available_tools()` filters out disabled tools so the agent never calls them
- [x] System prompt includes explicit fallback mapping and retry guidance
- [x] Startup banner shows tool status (available vs disabled with fallback info)
- [x] Fail-fast on missing `ANTHROPIC_API_KEY` with clear setup instructions

## Medium Impact

### 6. ~~Web UI~~ (DONE)
- [x] Streamlit chat interface with `st.chat_message` / `st.chat_input`
- [x] Real-time streaming feedback (thinking, tool calls) via `st.status` widget
- [x] Sidebar: tool health status, session save/load/clear, query timing
- [x] Run with `streamlit run app.py`

### 7. ~~Observability~~ (DONE)
- [x] `ObservabilityCallbackHandler` captures token usage, tool metrics, and timing per query
- [x] `QueryMetrics` dataclass with cost calculation (model-aware pricing tables)
- [x] `MetricsStore` persists metrics to `observability/metrics.jsonl` (JSONL append-only)
- [x] Streamlit dashboard: token counts, cost, tool calls, performance history with bar chart
- [x] CLI: metrics printed after each query + `stats` command for aggregate stats

### 8. ~~Async / True Parallelism~~ (DONE)
- [x] All I/O replaced with native async: `aiohttp` for HTTP, `asyncio.to_thread()` for blocking libraries
- [x] Agent uses `ainvoke()`/`astream()` (LangGraph's native async API)
- [x] `parallel_tool` uses `asyncio.gather()` instead of ThreadPoolExecutor
- [x] `async_retry_on_error` with `asyncio.sleep()`, `async_run_with_timeout` with `asyncio.wait_for()`
- [x] All 20 tools have `coroutine=` for native async execution
- [x] CLI and Streamlit use `asyncio.run()` entry points

## Nice to Have

### 9. ~~Config Cleanup~~ (DONE)
- [x] Make model selectable (currently hardcoded to claude-sonnet-4-5-20250929)
- [x] Fix banner to list all tools

### 10. ~~Rate Limiting~~ (DONE)
- [x] `RateLimiter` class with per-session token budget enforcement
- [x] Disabled by default — enable and set budget from Streamlit UI in real-time
- [x] Progress bar, warnings at 80%, error when exhausted
- [x] Resets on session clear

### 11. ~~Session Schema Versioning~~ (DONE)
- [x] Added `version: "1.0"` field to session JSON format
- [x] Backward compatible — old sessions without version field still load

---

## Next Level — From Good to Exceptional

These improvements would elevate the project from a strong portfolio piece to an outstanding demonstration of agent engineering depth.

### 12. ~~Plan-and-Execute Loop~~ (DONE)
- [x] `src/planner.py`: `ResearchStep` / `ResearchPlan` Pydantic models + `generate_plan()` + complexity detector (`is_simple_query()`)
- [x] `ResearchAgent._build_plan_execute_graph()`: LangGraph StateGraph with `create_plan → execute_step → replan → synthesize` nodes; conditional edge loops until all steps are done
- [x] `ResearchAgent.plan_and_execute()`: async method for CLI — prints each step, streams synthesis token-by-token
- [x] `ResearchAgent.plan_and_execute_stream()`: sync generator for Streamlit — yields typed events (`plan_created`, `step_started`, `step_tool`, `step_done`, `synthesis_token`, `done`)
- [x] Web UI: **Research Mode** sidebar toggle (Auto / Direct / Plan-and-Execute); live plan panel shows ⏳/🔄/✅ per step; synthesis streams word-by-word
- [x] CLI: `--plan` flag forces plan-and-execute; plan steps printed before execution
- [x] Auto mode uses `is_simple_query()` heuristic to choose between direct and plan modes
- [x] Backward compatible — `query()` and `stream_query()` untouched

### 13. Multi-Agent Orchestration
- [ ] Implement a supervisor/worker pattern (e.g., orchestrator agent delegates to specialist agents)
- [ ] Add a researcher + fact-checker dual-agent flow (one gathers, one verifies)
- [ ] Use LangGraph's multi-agent primitives for agent-to-agent communication
- [ ] Show agents with different tool subsets and system prompts collaborating on a task

### 14. RAG Pipeline & Semantic Memory
- [ ] Add a vector store (e.g., ChromaDB or FAISS) for embedding-based retrieval
- [ ] Store past research results and retrieved documents as embeddings
- [ ] Let the agent query semantic memory before making external API calls
- [ ] Support document ingestion (upload PDFs/URLs → chunk → embed → store)
- [ ] Demonstrate long-term knowledge accumulation across sessions

### 15. Human-in-the-Loop Interaction
- [ ] Add a mechanism for the agent to ask clarifying questions before proceeding
- [ ] Present intermediate results for user approval at key decision points
- [ ] Support user corrections mid-workflow (e.g., "focus more on X, skip Y")
- [ ] Implement confidence-based escalation — agent asks for help when uncertain

### 16. Deployment & Production Readiness
- [ ] Add a FastAPI wrapper exposing the agent as a REST API
- [ ] Dockerfile and docker-compose for containerized deployment
- [ ] CI/CD pipeline (GitHub Actions) for tests, linting, and build validation
- [ ] Environment-based configuration (dev/staging/prod)
- [ ] API authentication and rate limiting at the endpoint level
