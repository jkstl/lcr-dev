# LCR Project Handoff Prompt

Use this prompt when starting a new session with a coding agent to continue work on the LCR project.

---

## Prompt for New Coding Agent

```
I'm working on "LCR - Local Cognitive RAG System", a privacy-first conversational AI with persistent memory. The repo is at: https://github.com/jkstl/lcr-dev

Please review:
1. **IMPLEMENTATION_STATUS.md** - Current state and BLOCKING ISSUE at the top
2. **CLAUDE.md** - Full technical specification (authoritative source of truth)
3. **README.md** - User-facing overview

CURRENT BLOCKING ISSUE:
The Observer component is configured to run on CPU (num_gpu=0) to prevent VRAM contention with the main LLM, but all Observer tasks are failing silently. Error messages are empty. Traceback logging was just added but not yet tested.

Key files for this issue:
- src/models/llm.py - OllamaClient with num_gpu parameter
- src/observer/observer.py - Uses num_gpu=0 for CPU offloading
- src/config.py - observer_model setting

The goal is to get the Observer working on CPU so it doesn't compete for VRAM with the main 14B model. The main LLM and reranker work fine on GPU.

Please help me debug and fix the Observer CPU offloading issue.
```

---

## Key Context for Agent

### What Works
- Main LLM (qwen3:14b) - generates responses correctly
- Reranker (BGE-Reranker-v2-m3) - improves retrieval quality
- LanceDB vector memory - stores/retrieves memories
- FalkorDB graph - entity/relationship tracking (requires Docker)

### What's Broken
- Observer (qwen3:4b with num_gpu=0) fails silently on all tasks:
  - Utility grading
  - Summary generation
  - Entity extraction
  - Query generation

### Suspected Causes
1. Ollama might not support `num_gpu=0` in the API payload
2. CPU inference might be too slow causing timeouts
3. Model loading on CPU might fail silently

### Files Changed in This Session
- `src/models/llm.py` - Added num_gpu parameter
- `src/observer/observer.py` - Uses num_gpu=0, added traceback logging
- `src/config.py` - Added observer_model setting
- `src/observer/prompts.py` - Added RELATED_TO flexible pattern
- `src/memory/context_assembler.py` - Integrated reranker
- `src/models/reranker.py` - NEW: BGE-Reranker wrapper

### Models Required
```bash
ollama pull qwen3:14b   # Main LLM
ollama pull qwen3:4b    # Observer
```
