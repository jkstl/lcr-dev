# LCR - Local Cognitive RAG System

A privacy-first conversational AI assistant with persistent episodic memory. Runs entirely locally without external API dependencies.

**Current Version:** v0.4.3
**Status:** Text-based chat interface with vector memory, asynchronous background processing, streaming output, graph-based entity tracking with contradiction resolution, cross-encoder reranking, and improved entity extraction for family/romantic relationships.

---

## Overview

LCR (Local Cognitive RAG) is a conversational AI system designed to maintain long-term episodic memory across sessions. The system stores and retrieves factual information about the user's work, relationships, preferences, and context to provide personalized responses.

### Key Capabilities

- **Persistent Memory**: Stores conversation history with semantic search capabilities
- **Entity Tracking**: Maintains a knowledge graph of people, places, organizations, and relationships
- **Contradiction Resolution**: Automatically supersedes outdated information when facts change
- **Smart Retrieval**: Combines vector similarity search with cross-encoder reranking for relevant context
- **Temporal Understanding**: Tracks state changes and evolving intentions over time

### Example Usage

After providing context about work schedule and commute details:

```
User: What time do I need to be at work tomorrow?
Assistant: You need to leave by 8:30. Your shift at Acme starts at 9, and with 
           your usual commute from Cedar Park, that gives you a small buffer.
```

---

## Quick Start

### Prerequisites

- **Ollama** - Local LLM runtime ([ollama.ai](https://ollama.ai))
- **Docker** - Optional, required for graph database features
- **Python 3.11 or higher**

### Installation

```bash
# Install required Ollama models
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# Clone repository and install dependencies
cd lcr
pip install -r requirements.txt

# Optional: Start FalkorDB for graph features
docker-compose up -d

# Run the assistant
python -m src.main
```

### First Run

```
╭──────────────────────────────────────────╮
│   LCR - Local Cognitive RAG System       │
│   A memory-enhanced AI assistant         │
│                                          │
│   Model: qwen3:14b                       │
│   Memory count: 0                        │
╰──────────────────────────────────────────╯

Type 'quit' or 'exit' to end the conversation.

You: Hi! My name is Alex and I work at TechCorp as a software engineer.
Assistant: Nice to meet you, Alex! It's great to have a software engineer 
           from TechCorp here. What kind of projects do you work on?
```

Conversation data is automatically persisted. Subsequent sessions will have access to previously shared information.

---

## Architecture

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Main LLM | Qwen3 14B | Conversation generation |
| Observer LLM | Qwen3 1.7B | Background memory extraction |
| Vector Store | LanceDB | Semantic memory storage |
| Graph Store | FalkorDB | Entity-relationship tracking |
| Reranker | BGE-Reranker-v2-m3 | Relevance scoring |
| Embeddings | nomic-embed-text | Vector generation |

### Project Structure

```
lcr/
├── CLAUDE.md                    # Complete technical specification
├── IMPLEMENTATION_STATUS.md     # Current development status
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # FalkorDB configuration
├── tests/                       # Test suite
│   └── test_observer_quality.py # Observer extraction tests
├── src/
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration management
│   ├── models/
│   │   ├── llm.py             # Ollama API client
│   │   ├── reranker.py        # Cross-encoder wrapper
│   │   └── embedder.py        # Embedding generation
│   ├── memory/
│   │   ├── vector_store.py    # LanceDB operations
│   │   ├── graph_store.py     # FalkorDB operations
│   │   └── context_assembler.py # Retrieval orchestration
│   └── observer/
│       ├── observer.py         # Async memory extraction
│       └── prompts.py          # Extraction templates
└── data/
    ├── lancedb/                # Vector database (auto-created)
    ├── falkordb/               # Graph database (auto-created)
    └── conversations/          # Conversation logs (auto-created)
```

---

## Configuration

Create `.env` from `.env.template` to customize settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `MAIN_MODEL` | `qwen3:14b` | Primary conversation model |
| `OBSERVER_MODEL` | `qwen3:1.7b` | Background extraction model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding generation model |
| `LANCEDB_PATH` | `./data/lancedb` | Vector storage location |
| `VECTOR_SEARCH_TOP_K` | `5` | Number of memories to retrieve |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |

### Hardware-Constrained Deployments

For systems with limited VRAM, consider using smaller models:

```bash
ollama pull qwen3:8b  # Requires ~6GB VRAM instead of ~10GB
```

Update `.env`:
```
MAIN_MODEL=qwen3:8b
```

---

## Current Status

### Implemented Features (v0.4.3)

- Vector-based memory storage with LanceDB
- Graph-based entity and relationship tracking with FalkorDB
- Asynchronous background memory processing (Observer)
- Cross-encoder reranking for improved retrieval
- Temporal intent and state change extraction
- Contradiction detection and resolution
- Streaming token-by-token output
- CLI interface with conversation persistence
- Family and romantic relationship extraction (SIBLING_OF, EX_PARTNER_OF, etc.)
- Improved extraction rules for attribution and temporal metadata

### Known Limitations

**Observer Processing**: Background memory extraction completes asynchronously. The system waits for pending tasks before shutdown, which may take several seconds depending on conversation complexity.

**Graph Database**: Entity tracking and contradiction detection require FalkorDB via Docker. Without it, the system operates in vector-only mode with reduced functionality.

---

## Development Roadmap

### Completed Phases
- Phase 1: Vector memory foundation
- Phase 2: Observer system
- Phase 3: Graph database integration
- Phase 3.1: Temporal relationship extraction
- Phase 4: Cross-encoder reranking
- Phase 4.1: Observer CPU optimization
- Phase 4.2: Observer extraction quality improvements

### Planned Phases
- Voice interface (Whisper STT + Piper TTS)
- LangGraph state machine orchestration
- Redis task queue for scalability

See `IMPLEMENTATION_STATUS.md` for detailed progress tracking.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| VRAM | 8GB | 16GB |
| System RAM | 16GB | 32GB |
| Storage | HDD | SSD |

---

## License

MIT License - See `CLAUDE.md` for complete architectural documentation.
