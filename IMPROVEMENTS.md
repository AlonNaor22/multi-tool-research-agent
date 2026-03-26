# Planned Improvements

Prioritized list of improvements to make this project stand out as a portfolio piece for agent engineering roles.

## High Impact

### 1. ~~Tests~~ (DONE)
- [x] 188 tests across all tools, memory, sessions, callbacks, and utilities

### 2. ~~Migrate to Native Tool Calling~~ (DONE)
- [x] Replaced `create_react_agent` (string-based ReAct parsing) with `create_agent` (LangGraph-based)
- [x] Uses Claude's native structured tool-use API instead of fragile text parsing
- [x] Messages-based conversation history instead of string concatenation
- [x] Updated requirements to langchain 1.x + langgraph

### 3. Evaluation Suite
- [ ] Create 10-20 curated questions with expected tool selections and answer keywords
- [ ] Score: did the agent pick the right tool? Did it answer correctly?
- [ ] Shows you think about agents as systems, not just demos

### 4. Streaming Output
- [ ] Stream Thought/Action/Observation steps in real-time instead of blank screen while thinking
- [ ] Shows production UX thinking

### 5. Better Error Recovery
- [ ] Tool health check on startup — show which tools are available vs disabled (missing API keys)
- [ ] Fallback to a different tool if one fails
- [ ] Retry with rephrased query on failure
- [ ] Graceful degradation when optional API keys are missing

## Medium Impact

### 6. Web UI
- [ ] Simple Streamlit or Gradio frontend (even 50 lines)
- [ ] Makes the project demo-able in interviews instead of CLI-only

### 7. Observability
- [ ] Token usage tracking (cost per query)
- [ ] Tool call success/failure rates
- [ ] Full trace logging (LangSmith or similar)

### 8. Async / True Parallelism
- [ ] Replace ThreadPoolExecutor workaround with native async (`ainvoke`)
- [ ] What production agents actually use

## Nice to Have

### 9. ~~Config Cleanup~~ (DONE)
- [x] Make model selectable (currently hardcoded to claude-sonnet-4-5-20250929)
- [x] Fix banner to list all 17 tools

### 10. Rate Limiting
- [ ] Prevent burning API credits if someone hammers the agent

### 11. Session Schema Versioning
- [ ] Add version field to session JSON for forward compatibility
