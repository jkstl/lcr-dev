# CLAUDE.md — Local Cognitive RAG (LCR) System
## Authoritative Architectural Blueprint v1.0

> **Purpose**: This document defines the complete technical specification for a local, privacy-first conversational AI assistant with persistent episodic memory. Any coding agent (Claude Code, Codex, Cursor) should be able to implement this system from this specification alone.

---

## IMPLEMENTATION STATUS (As of 2026-01-16)

### ✅ Phase 1: MVP Complete
- Vector memory with LanceDB
- Ollama integration (Qwen3:14b + nomic-embed-text)
- Basic CLI with Rich interface
- Conversation persistence

### ✅ Phase 2: Observer System Complete
- Async utility grading (DISCARD/LOW/MEDIUM/HIGH)
- Entity/relationship extraction
- Summary generation
- Retrieval query pre-generation
- **Fire-and-forget processing** (30-60s delay before memories persist)

### ✅ Phase 3: Graph Database Complete
- FalkorDB integration via Docker
- Person & Entity nodes in graph
- Relationship tracking (WORKS_AT, KNOWS, OWNS, etc.)
- **Contradiction detection & supersession** (marks old facts as `still_valid: false`)
- Graph-enhanced context assembly

### 🔄 Phase 4: Cross-Encoder Reranking (Planned)
- BGE-Reranker-v2-m3 for better memory relevance
- Two-stage retrieval: vector search → rerank → LLM

### ⏸️ Deferred Features
- Voice interface (Whisper STT + Piper TTS)
- LangGraph orchestration
- Redis task queue
- Cross-encoder reranking

### 🐛 Known Issues
1. **Observer Delay**: Async processing takes 30-60s. Users must wait before exiting or memories aren't stored.
2. **Docker Setup**: FalkorDB requires Docker Compose V2 (`/usr/local/bin/docker-compose` on Ubuntu 24.04 due to distutils compatibility)
3. **Model**: Currently uses `qwen3:14b` (newer than spec's `qwen2.5:14b`)

---

## 1. PROJECT OVERVIEW

### 1.1 What This System Does
A voice-enabled conversational AI that:
- Remembers everything the user tells it across sessions (work schedules, relationships, owned technology, preferences)
- Retrieves relevant memories automatically when answering questions
- Learns the user's personality, communication style, and life context over time
- Runs 100% locally with zero external API calls

### 1.2 Example Interactions After Training
```
User: "What time do I need to be at work tomorrow?"
System: [Retrieves: User works at Acme Corp, shift starts 9am, 30min commute]
Response: "You need to leave by 8:30. Your shift at Acme starts at 9, and with your usual commute from Cedar Park, that gives you a small buffer."

User: "Should I text Sarah back?"
System: [Retrieves: Recent tension with Sarah, user expressed feeling overwhelmed, Sarah's communication pattern]
Response: "Based on what you've shared, you were feeling pretty drained after your last exchange. If you're not ready, it's okay to wait until tomorrow. What's making you hesitate?"
```

### 1.3 Hardware Constraints
| Resource | Limit | Allocation Strategy |
|----------|-------|---------------------|
| VRAM | 16GB | Main LLM ~10GB, Reranker ~0.5GB, Buffer ~5.5GB |
| System RAM | 32GB | Vector DB, Graph DB, Application |
| Storage | SSD Required | LanceDB disk-persistence, conversation logs |
| Network | None | 100% air-gapped capable |

---

## 2. TECHNOLOGY STACK

### 2.1 Core Components

| Component | Technology | Version/Variant | Resource Placement |
|-----------|------------|-----------------|-------------------|
| **Runtime** | Python | 3.11+ | CPU |
| **Main LLM** | Qwen 3 Instruct | 14B Q4_K_M (GGUF) | VRAM (~10GB) |
| **Observer LLM** | Phi-3.5 Mini Instruct | 3.8B Q4_K_M | CPU (offloaded) |
| **Embedding Model** | nomic-embed-text | v1.5 | CPU |
| **Reranker** | BGE-Reranker | v2-m3 | VRAM (~0.5GB) |
| **Vector Database** | LanceDB | Latest | Disk + RAM cache |
| **Graph Database** | FalkorDB | Latest | Docker container |
| **Orchestration** | LangGraph | Latest | CPU |
| **LLM Backend** | Ollama | Latest | Manages GGUF models |
| **Voice Input** | Whisper.cpp | Medium model | CPU |
| **Voice Output** | Piper TTS | en_US-lessac-medium | CPU |

### 2.2 Python Dependencies
```
# requirements.txt (CURRENT as of v0.3.0)
ollama>=0.3.0
lancedb>=0.6.0
falkordb>=1.0.0
numpy>=1.26.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
rich>=13.0.0
httpx>=0.27.0

# NOT YET IMPLEMENTED:
# sentence-transformers>=2.7.0  # For reranker
# langgraph>=0.1.0
# redis>=5.0.0
# faster-whisper>=1.0.0
# piper-tts>=1.0.0
# sounddevice>=0.4.6
```

### 2.3 Ollama Model Setup
```bash
# Install models (run once)
ollama pull qwen3:14b              # Currently using qwen3 instead of qwen2.5
ollama pull nomic-embed-text

# NOT YET USED:
# ollama pull phi3.5:3.8b-mini-instruct-q4_K_M  # For Observer (future optimization)
```

### 2.4 Docker Services
```yaml
# docker-compose.yml (CURRENT)
version: '3.8'
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - ./data/falkordb:/data
    command: ["--save", "60", "1"]  # Persist every 60s if 1+ change

# NOT YET IMPLEMENTED:
#   redis:
#     image: redis:7-alpine
#     ports:
#       - "6380:6379"
```

---

*[Rest of CLAUDE.md continues unchanged...]*
