# LCR - Local Cognitive RAG System

A voice-enabled, privacy-first conversational AI assistant with persistent episodic memory. Runs 100% locally with zero external API calls.

> **Current Status: v0.3.0**  
> Text-only chat with vector memory, async Observer, streaming output, and FalkorDB graph for entity tracking & contradiction resolution.

## 🎯 What This Does

LCR remembers everything you tell it across sessions:
- Work schedules, relationships, owned technology, preferences
- Retrieves relevant memories automatically when answering questions
- Learns your personality and life context over time

**Example after training:**
```
You: What time do I need to be at work tomorrow?
LCR: You need to leave by 8:30. Your shift at Acme starts at 9, and with 
     your usual commute from Cedar Park, that gives you a small buffer.
```

## 🚀 Quick Start

### Prerequisites

1. **Ollama** - Install from [ollama.ai](https://ollama.ai)
2. **Docker** (optional) - For graph database features
3. **Python 3.11+**

### Setup

```bash
# 1. Install Ollama models
ollama pull qwen3:14b
ollama pull nomic-embed-text

# 2. Clone and install dependencies
cd lcr
pip install -r requirements.txt

# 3. (Optional) Start FalkorDB for graph features
docker-compose up -d

# 4. Run the assistant
python -m src.main
```

## Known Issues

> **⚠️ Important:** The Observer system processes memories in the background (~30-60s). Wait at least 30 seconds after your last message before typing `exit`, otherwise memories won't be stored.

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

You: quit
Conversation saved to: data/conversations/abc123.json
```

The next time you start a conversation, LCR will remember who you are!

## 📁 Project Structure

```
lcr/
├── CLAUDE.md              # Full architecture spec
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env.template          # Config template
├── src/
│   ├── __init__.py
│   ├── main.py            # CLI entry point
│   ├── config.py          # Settings management
│   ├── models/
│   │   ├── llm.py         # Ollama client
│   │   └── embedder.py    # Embedding generation
│   ├── memory/
│   │   ├── vector_store.py      # LanceDB storage
│   │   └── context_assembler.py # Memory retrieval
│   └── observer/
│       ├── observer.py    # Memory extraction & grading
│       └── prompts.py     # Extraction prompts
└── data/
    ├── lancedb/           # Vector database (auto-created)
    └── conversations/     # Conversation logs (auto-created)
```

## ⚙️ Configuration

Copy `.env.template` to `.env` and customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `MAIN_MODEL` | `qwen3:14b` | Main LLM model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LANCEDB_PATH` | `./data/lancedb` | Vector store location |
| `VECTOR_SEARCH_TOP_K` | `5` | Memories to retrieve |

### Using Smaller Models

For less powerful hardware, use a smaller model:

```bash
ollama pull qwen3:8b
```

Then set in `.env`:
```
MAIN_MODEL=qwen3:8b
```

## 🔮 Roadmap

### Current
- [x] Basic chat with memory
- [x] LanceDB vector storage
- [x] Conversation persistence
- [x] Rich CLI interface
- [x] **Observer System** - Utility grading, entity extraction, smart summaries
- [x] **Streaming Output** - Token-by-token response
- [x] **Graph Database (FalkorDB)** - Entity-relationship tracking & contradiction resolution
- [x] **Graph-Enhanced Context** - Retrieves entity facts from graph for richer responses

### Coming Soon
- [ ] **Cross-encoder Reranking**: Better memory relevance
- [ ] **Voice Interface**: Whisper STT + Piper TTS
- [ ] **LangGraph Orchestration**: Advanced state machine

## 🏗️ Hardware Requirements

| Component | MVP | Full Version |
|-----------|-----|--------------|
| VRAM | 8GB+ | 16GB |
| RAM | 16GB | 32GB |
| Storage | SSD recommended | SSD required |

## 📜 License

MIT License - See CLAUDE.md for full architecture documentation.

---

*Built with ❤️ for local-first AI*
