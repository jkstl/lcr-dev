# LCR Implementation Status

> **Last Updated**: 2026-01-17  
> **Current Version**: v0.4.1  
> **Repository**: https://github.com/jkstl/lcr-dev

---

## 🚨 CURRENT BLOCKING ISSUE

### Observer CPU Offloading Failure

**Status**: RESOLVED (v0.4.2)
**Priority**: Normal

The Observer component previously failed when running on CPU (`num_gpu=0`) due to Ollama hanging when receiving concurrent requests.
 
**Resolution**:
- **Sequential Execution**: Modified `src/observer/observer.py` to execute extraction tasks sequentially instead of in parallel.
- **Model Optimization**: Default Observer model switched to `qwen3:0.6b` for lightweight CPU performance.
- **Enhanced Logging**: Added detailed error logging to `src/models/llm.py`.
 
**Verification**:
- Validated with `reproduce_issue.py` (now deleted).
- Confirmed stable execution on CPU.


---

## ✅ Completed Phases

### Phase 1: MVP (v0.1.0)
- [x] Vector memory with LanceDB
- [x] Ollama integration (Qwen3:14b + nomic-embed-text)
- [x] Basic CLI with Rich interface
- [x] Conversation persistence to JSON

### Phase 2: Observer System (v0.2.0)
- [x] Async utility grading (DISCARD/LOW/MEDIUM/HIGH)
- [x] Entity/relationship extraction
- [x] Summary generation
- [x] Retrieval query pre-generation
- [x] Fire-and-forget background processing

### Phase 3: Graph Database (v0.3.0)
- [x] FalkorDB integration via Docker
- [x] Person & Entity nodes in graph
- [x] Relationship tracking (OWNS, WORKS_AT, KNOWS, etc.)
- [x] Contradiction detection & supersession
- [x] Graph-enhanced context assembly

### Phase 3.1: Temporal Entity Extraction (v0.3.1)
- [x] Added temporal relationship types (PLANS_TO, CURRENTLY_USING_AS, IN_STATE, WAS_IN_STATE)
- [x] Enhanced extraction prompt with examples
- [x] Added RELATED_TO flexible pattern for domain-specific relationships

### Phase 4: Cross-Encoder Reranking (v0.4.0)
- [x] BGE-Reranker-v2-m3 integration
- [x] Two-stage retrieval: vector (top-15) → rerank (top-5)
- [x] GPU/CPU auto-detection with fallback
- [x] Added `sentence-transformers` dependency

### Phase 4.1: Observer Model Separation (v0.4.1) - IN PROGRESS
- [x] Added `observer_model` config setting (qwen3:4b)
- [x] Added `num_gpu` parameter to OllamaClient
- [x] Observer uses CPU offloading (`num_gpu=0`)
- [ ] **BLOCKED**: Observer fails silently on CPU (see above)

---

## ⏸️ Deferred Features

- Voice interface (Whisper STT + Piper TTS)
- LangGraph orchestration
- Redis task queue

---

## 🏗️ Architecture Overview

### Resource Allocation (Target)
| Component | Device | VRAM/RAM |
|-----------|--------|----------|
| Main LLM (qwen3:14b) | GPU | ~10GB VRAM |
| Reranker (BGE-v2-m3) | GPU | ~2.1GB VRAM |
| Observer (qwen3:4b) | **CPU** | ~3GB RAM |
| LanceDB | CPU | RAM cache |
| FalkorDB | Docker | Disk + RAM |

### Key Files
| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point, LCRAssistant class |
| `src/config.py` | Pydantic settings (models, paths, etc.) |
| `src/models/llm.py` | OllamaClient with `num_gpu` support |
| `src/models/reranker.py` | BGE-Reranker-v2-m3 wrapper |
| `src/observer/observer.py` | Background memory extraction |
| `src/observer/prompts.py` | Extraction prompt templates |
| `src/memory/vector_store.py` | LanceDB operations |
| `src/memory/graph_store.py` | FalkorDB operations |
| `src/memory/context_assembler.py` | Retrieval + reranking |

---

## 📋 Testing Checklist

Before testing, ensure:
```bash
# Required models
ollama pull qwen3:14b
ollama pull qwen3:4b

# Optional: FalkorDB for graph features
docker compose up -d

# Install dependencies
pip install -r requirements.txt
```

Run: `python -m src.main`

---

## 📚 Reference Documents

- **CLAUDE.md** - Full technical specification (authoritative)
- **README.md** - User-facing overview and quick start
- **requirements.txt** - Python dependencies
