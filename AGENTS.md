# CLAUDE.md — Local Cognitive RAG (LCR) System
## Authoritative Architectural Blueprint v1.0

> **Purpose**: This document defines the complete technical specification for a local, privacy-first conversational AI assistant with persistent episodic memory. Any coding agent (Claude Code, Codex, Cursor) should be able to implement this system from this specification alone.

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
| **Main LLM** | Qwen3 | 14B Q4_K_M (GGUF) | VRAM (~10GB) |
| **Observer LLM** | Qwen3:1.7b | 4B Q4_K_M | CPU (offloaded) |
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
# requirements.txt
ollama>=0.2.0
langgraph>=0.1.0
lancedb>=0.6.0
falkordb>=1.0.0
sentence-transformers>=2.7.0  # For reranker
numpy>=1.26.0
pydantic>=2.0.0
redis>=5.0.0                  # For async task queue
faster-whisper>=1.0.0         # Whisper.cpp Python bindings
piper-tts>=1.0.0
sounddevice>=0.4.6            # Audio I/O
httpx>=0.27.0                 # Async HTTP for Ollama
rich>=13.0.0                  # CLI interface
python-dotenv>=1.0.0
```

### 2.3 Ollama Model Setup
```bash
# Install models (run once)
ollama pull qwen3:14b
ollama pull qwen3:1.7b
ollama pull nomic-embed-text:v1.5
```

### 2.4 Docker Services
```yaml
# docker-compose.yml
version: '3.8'
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - ./data/falkordb:/data
    command: ["--save", "60", "1"]  # Persist every 60s if 1+ change
  
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - ./data/redis:/data
```

**IMPLEMENTATION NOTE (v0.4.0 - Phase 4 Complete)**: Cross-encoder reranking has been implemented using BGE-Reranker-v2-m3. The system now uses two-stage retrieval: (1) vector search retrieves top-15 candidates, (2) cross-encoder reranks and selects top-5 most relevant. This is configurable via `settings.use_reranker` and can be disabled for fallback to vector-only retrieval. See `src/models/reranker.py` for implementation details. The reranker auto-detects GPU/CPU and gracefully degrades to CPU if VRAM is unavailable.

**IMPLEMENTATION NOTE (v0.4.1)**: Observer now uses separate `qwen3:1.7b` model (configurable via `observer_model` setting) instead of main 14B model, significantly improving Observer processing speed. Also added `RELATED_TO` flexible pattern in extraction prompt for domain-specific relationships (e.g., infrastructure: connected_to, depends_on) without hardcoding each type.

---

## 3. DATA ARCHITECTURE

### 3.1 Directory Structure
```
lcr/
├── CLAUDE.md                    # This file
├── requirements.txt
├── docker-compose.yml
├── .env                         # Local config (not committed)
├── src/
│   ├── __init__.py
│   ├── main.py                  # Entry point
│   ├── config.py                # Pydantic settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py               # Ollama client wrapper
│   │   ├── embedder.py          # Embedding generation
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # LanceDB operations
│   │   ├── graph_store.py       # FalkorDB operations
│   │   └── context_assembler.py # Retrieval orchestration
│   ├── observer/
│   │   ├── __init__.py
│   │   ├── observer.py          # Main observer logic
│   │   ├── extractors.py        # Entity/relationship extraction
│   │   └── prompts.py           # Observer prompt templates
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py           # Semantic chunking
│   │   └── pipeline.py          # Document ingestion
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py               # Speech-to-text (Whisper)
│   │   └── tts.py               # Text-to-speech (Piper)
│   └── orchestration/
│       ├── __init__.py
│       └── graph.py             # LangGraph state machine
├── data/
│   ├── lancedb/                 # Vector DB storage
│   ├── falkordb/                # Graph DB storage
│   ├── redis/                   # Task queue storage
│   └── conversations/           # Raw conversation logs (JSON)
└── tests/
    └── ...
```

### 3.2 LanceDB Schema (Vector Store)

```python
# memory/vector_store.py

import lancedb
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MemoryChunk(BaseModel):
    """Schema for vector store entries."""
    id: str                          # UUID
    content: str                     # The actual text
    summary: str                     # Observer-generated 1-sentence summary
    embedding: list[float]           # 768-dim nomic-embed-text vector
    
    # Metadata
    chunk_type: str                  # "conversation" | "document" | "fact"
    source_conversation_id: str      # Links to raw conversation log
    turn_index: int                  # Position in conversation
    
    # Temporal
    created_at: datetime
    last_accessed_at: datetime
    access_count: int                # For relevance boosting
    
    # Pre-generated retrieval queries
    retrieval_queries: list[str]     # 2-3 questions this chunk answers
    
    # Utility
    utility_score: float             # 0.0-1.0, from Observer grading
    
    class Config:
        # LanceDB will auto-create vector index on 'embedding' field
        pass

# Table creation
def init_vector_store(db_path: str = "./data/lancedb"):
    db = lancedb.connect(db_path)
    
    # Create table if not exists
    if "memories" not in db.table_names():
        db.create_table("memories", schema=MemoryChunk)
    
    return db.open_table("memories")
```

### 3.3 FalkorDB Schema (Knowledge Graph)

```cypher
// Node Types

// Person nodes (user and people they mention)
CREATE (p:Person {
    id: "uuid",
    name: "string",
    relationship_to_user: "string",  // "self" | "partner" | "friend" | "coworker" | "family"
    first_mentioned: datetime,
    last_mentioned: datetime
})

// Entity nodes (things, places, concepts)
CREATE (e:Entity {
    id: "uuid",
    name: "string",
    category: "string",  // "technology" | "place" | "organization" | "project" | "concept"
    attributes: map,     // Flexible key-value for entity-specific data
    first_mentioned: datetime,
    last_mentioned: datetime,
    still_valid: boolean // For temporal accuracy
})

// Fact nodes (discrete pieces of information)
CREATE (f:Fact {
    id: "uuid",
    statement: "string",
    confidence: float,   // 0.0-1.0
    source_conversation_id: "string",
    created_at: datetime,
    superseded_by: "uuid | null",  // For contradiction resolution
    still_valid: boolean
})

// Edge Types (Relationships)

// Person relationships
(Person)-[:KNOWS {since: datetime, context: string}]->(Person)
(Person)-[:WORKS_AT {role: string, started: datetime, ended: datetime | null}]->(Entity:Organization)
(Person)-[:LIVES_IN {since: datetime}]->(Entity:Place)
(Person)-[:OWNS {acquired: datetime}]->(Entity:Technology)

// User-specific edges
(Person:User)-[:FEELS_ABOUT {sentiment: string, intensity: float, timestamp: datetime}]->(Person|Entity)
(Person:User)-[:PREFERS {context: string, strength: float}]->(Entity)
(Person:User)-[:SCHEDULED {time: string, recurrence: string, valid_until: datetime}]->(Entity:Event)

// Entity relationships
(Entity)-[:PART_OF]->(Entity)
(Entity)-[:LOCATED_IN]->(Entity:Place)
(Entity)-[:RELATED_TO {relationship_type: string}]->(Entity)

// Fact connections
(Fact)-[:ABOUT]->(Person|Entity)
(Fact)-[:SUPERSEDES]->(Fact)  // For contradiction chains
```

### 3.4 Conversation Log Schema

```python
# Stored in data/conversations/{conversation_id}.json

from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    role: str                    # "user" | "assistant"
    content: str
    timestamp: datetime
    
    # Observer outputs (populated async)
    observer_processed: bool = False
    utility_score: float | None = None
    extracted_entities: list[str] = []
    extracted_relationships: list[dict] = []
    summary: str | None = None

class Conversation(BaseModel):
    id: str                      # UUID
    started_at: datetime
    ended_at: datetime | None
    messages: list[Message]
    
    # Metadata
    total_turns: int
    topics_discussed: list[str]  # High-level topic tags
```

---

## 4. CORE ALGORITHMS

### 4.1 Semantic Chunking Algorithm

```python
# ingestion/chunker.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Generator

class SemanticChunker:
    """
    Splits text into semantically coherent chunks based on embedding distance.
    """
    
    def __init__(
        self,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        similarity_threshold: float = 0.85,  # Cosine similarity; chunk breaks below this
        min_chunk_sentences: int = 3,
        max_chunk_tokens: int = 512
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_tokens = max_chunk_tokens
    
    def chunk(self, text: str) -> Generator[str, None, None]:
        """
        Algorithm:
        1. Split text into sentences
        2. Embed each sentence
        3. Calculate cosine similarity between consecutive sentences
        4. Break chunk when similarity drops below threshold
        5. Respect min/max constraints
        """
        sentences = self._split_sentences(text)
        if len(sentences) == 0:
            return
        
        embeddings = self.embedder.encode(sentences)
        
        current_chunk = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            sentence_tokens = self._count_tokens(sentences[i])
            
            # Decide whether to break
            should_break = (
                similarity < self.similarity_threshold
                and len(current_chunk) >= self.min_chunk_sentences
            )
            
            would_exceed_max = (current_tokens + sentence_tokens) > self.max_chunk_tokens
            
            if should_break or would_exceed_max:
                yield " ".join(current_chunk)
                current_chunk = [sentences[i]]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentences[i])
                current_tokens += sentence_tokens
        
        # Yield final chunk
        if current_chunk:
            yield " ".join(current_chunk)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _split_sentences(self, text: str) -> list[str]:
        # Use simple regex or spaCy for production
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    def _count_tokens(self, text: str) -> int:
        # Approximate: 1 token ≈ 4 characters for English
        return len(text) // 4
```

### 4.2 Context Assembly Algorithm

```python
# memory/context_assembler.py

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RetrievedContext:
    content: str
    source: str          # "vector" | "graph" | "sliding_window"
    relevance_score: float
    temporal_score: float
    final_score: float

class ContextAssembler:
    """
    Assembles context for the main LLM from multiple sources.
    Target: < 3000 tokens of highly relevant context.
    """
    
    def __init__(
        self,
        vector_store,      # LanceDB table
        graph_store,       # FalkorDB client
        reranker,          # Cross-encoder model
        max_context_tokens: int = 3000,
        sliding_window_tokens: int = 2000,
        temporal_decay_days: int = 30  # Half-life for temporal decay
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker
        self.max_context_tokens = max_context_tokens
        self.sliding_window_tokens = sliding_window_tokens
        self.temporal_decay_days = temporal_decay_days
    
    async def assemble(
        self,
        query: str,
        conversation_history: list[dict],  # Current conversation
        top_k_vector: int = 15,
        top_k_graph: int = 10,
        final_k: int = 5
    ) -> str:
        """
        Assembly pipeline:
        1. Include sliding window (most recent turns)
        2. Vector similarity search
        3. Graph entity lookup
        4. Merge and deduplicate
        5. Temporal decay weighting
        6. Cross-encoder reranking
        7. Format for LLM consumption
        """
        
        # Step 1: Sliding window
        sliding_context = self._get_sliding_window(conversation_history)
        remaining_tokens = self.max_context_tokens - self._count_tokens(sliding_context)
        
        # Step 2: Vector search
        vector_results = await self._vector_search(query, top_k=top_k_vector)
        
        # Step 3: Graph search
        entities = self._extract_entities_from_query(query)
        graph_results = await self._graph_search(entities, top_k=top_k_graph)
        
        # Step 4: Merge results
        all_candidates = self._merge_results(vector_results, graph_results)
        
        # Step 5: Apply temporal decay
        for candidate in all_candidates:
            candidate.temporal_score = self._calculate_temporal_decay(candidate.created_at)
            candidate.final_score = candidate.relevance_score * candidate.temporal_score
        
        # Step 6: Rerank with cross-encoder
        reranked = self._rerank(query, all_candidates, top_k=final_k)
        
        # Step 7: Format context
        memory_context = self._format_memories(reranked, max_tokens=remaining_tokens)
        
        return self._build_final_context(sliding_context, memory_context)
    
    def _calculate_temporal_decay(self, created_at: datetime) -> float:
        """Exponential decay with configurable half-life."""
        age_days = (datetime.now() - created_at).days
        return 0.5 ** (age_days / self.temporal_decay_days)
    
    def _get_sliding_window(self, history: list[dict]) -> str:
        """Get most recent conversation turns up to token limit."""
        result = []
        tokens = 0
        for msg in reversed(history):
            msg_tokens = self._count_tokens(msg["content"])
            if tokens + msg_tokens > self.sliding_window_tokens:
                break
            result.insert(0, f"{msg['role'].upper()}: {msg['content']}")
            tokens += msg_tokens
        return "\n".join(result)
    
    async def _vector_search(self, query: str, top_k: int) -> list[RetrievedContext]:
        """
        Search vector store with query embedding.
        Also searches against pre-generated retrieval_queries field.
        """
        # Implementation uses LanceDB's vector search
        pass
    
    async def _graph_search(self, entities: list[str], top_k: int) -> list[RetrievedContext]:
        """
        Query graph for facts about mentioned entities.
        Returns connected facts, relationships, and related entities.
        """
        # Cypher query implementation
        pass
    
    def _rerank(self, query: str, candidates: list[RetrievedContext], top_k: int) -> list[RetrievedContext]:
        """Use cross-encoder to rerank candidates."""
        pairs = [(query, c.content) for c in candidates]
        scores = self.reranker.predict(pairs)
        
        for candidate, score in zip(candidates, scores):
            candidate.final_score *= score  # Combine with temporal-weighted score
        
        return sorted(candidates, key=lambda x: x.final_score, reverse=True)[:top_k]
    
    def _build_final_context(self, sliding: str, memories: str) -> str:
        return f"""## Recent Conversation
{sliding}

## Relevant Memories
{memories}"""
```

### 4.3 Observer Algorithm

```python
# observer/observer.py

import asyncio
from dataclasses import dataclass
from enum import Enum

class UtilityGrade(Enum):
    DISCARD = "discard"      # "Thanks", "OK", small talk
    LOW = "low"              # General discussion, no new facts
    MEDIUM = "medium"        # Contains preferences or opinions
    HIGH = "high"            # Contains facts, schedules, relationships

@dataclass
class ObserverOutput:
    utility_grade: UtilityGrade
    summary: str | None
    entities: list[dict]           # [{name, type, attributes}]
    relationships: list[dict]      # [{subject, predicate, object, metadata}]
    contradictions: list[dict]     # [{existing_fact_id, new_statement, resolution_needed}]
    retrieval_queries: list[str]   # Pre-generated questions

class Observer:
    """
    Async post-generation reflection system.
    Runs after every assistant response to maintain memory.
    """
    
    def __init__(
        self,
        llm_client,           # Ollama client for qwen3:1.7b
        vector_store,
        graph_store,
        model: str = "qwen3:1.7b"
    ):
        self.llm = llm_client
        self.model = model
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    async def process_turn(
        self,
        user_message: str,
        assistant_response: str,
        conversation_id: str,
        turn_index: int
    ) -> ObserverOutput:
        """
        Main observer pipeline. Called async after each assistant response.
        """
        
        combined_input = f"USER: {user_message}\nASSISTANT: {assistant_response}"
        
        # Run extraction tasks in parallel
        utility_task = self._grade_utility(combined_input)
        extraction_task = self._extract_entities_and_relations(combined_input)
        summary_task = self._generate_summary(combined_input)
        query_task = self._generate_retrieval_queries(combined_input)
        
        utility_grade, extractions, summary, queries = await asyncio.gather(
            utility_task, extraction_task, summary_task, query_task
        )
        
        # Early exit if not worth remembering
        if utility_grade == UtilityGrade.DISCARD:
            return ObserverOutput(
                utility_grade=utility_grade,
                summary=None,
                entities=[],
                relationships=[],
                contradictions=[],
                retrieval_queries=[]
            )
        
        # Check for contradictions
        contradictions = await self._check_contradictions(extractions["relationships"])
        
        # Persist to stores
        await self._persist_to_vector_store(
            content=combined_input,
            summary=summary,
            queries=queries,
            utility_score=self._utility_to_score(utility_grade),
            conversation_id=conversation_id,
            turn_index=turn_index
        )
        
        await self._persist_to_graph_store(
            entities=extractions["entities"],
            relationships=extractions["relationships"],
            contradictions=contradictions
        )
        
        return ObserverOutput(
            utility_grade=utility_grade,
            summary=summary,
            entities=extractions["entities"],
            relationships=extractions["relationships"],
            contradictions=contradictions,
            retrieval_queries=queries
        )
    
    async def _grade_utility(self, text: str) -> UtilityGrade:
        """Determine if this turn is worth remembering."""
        prompt = """Rate the memory-worthiness of this conversation turn.

TURN:
{text}

Rules:
- DISCARD: Greetings, thanks, acknowledgments, small talk with no information
- LOW: General discussion but no concrete facts about the user
- MEDIUM: Contains user preferences, opinions, or feelings
- HIGH: Contains facts (schedules, relationships, owned items, work info)

Respond with exactly one word: DISCARD, LOW, MEDIUM, or HIGH"""
        
        response = await self.llm.generate(self.model, prompt.format(text=text))
        return UtilityGrade(response.strip().lower())
    
    async def _extract_entities_and_relations(self, text: str) -> dict:
        """Extract structured data from conversation turn."""
        prompt = """Extract entities and relationships from this conversation.

TURN:
{text}

Output valid JSON only:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "relationship type", "object": "entity2", "metadata": {{}}}}
    ]
}}

Common relationship types: WORKS_AT, LIVES_IN, OWNS, KNOWS, FEELS_ABOUT, PREFERS, SCHEDULED, MARRIED_TO, FRIENDS_WITH

Only extract what is explicitly stated. Do not infer."""
        
        response = await self.llm.generate(self.model, prompt.format(text=text))
        return self._parse_json(response)
    
    async def _generate_summary(self, text: str) -> str:
        """Generate 1-sentence summary for episodic search."""
        prompt = """Summarize this conversation turn in exactly one sentence.
Focus on what was discussed or decided, from the user's perspective.

TURN:
{text}

ONE SENTENCE SUMMARY:"""
        
        response = await self.llm.generate(self.model, prompt.format(text=text))
        return response.strip()
    
    async def _generate_retrieval_queries(self, text: str) -> list[str]:
        """Pre-generate questions this memory could answer."""
        prompt = """What questions could this conversation turn answer in the future?
Generate 2-3 natural questions a user might ask.

TURN:
{text}

Output as JSON array of strings:
["question 1", "question 2", "question 3"]"""
        
        response = await self.llm.generate(self.model, prompt.format(text=text))
        return self._parse_json(response)
    
    async def _check_contradictions(self, new_relationships: list[dict]) -> list[dict]:
        """Check if new facts contradict existing graph data."""
        contradictions = []
        
        for rel in new_relationships:
            # Query graph for existing facts about this subject-predicate pair
            existing = await self.graph_store.query(f"""
                MATCH (s {{name: $subject}})-[r:{rel['predicate']}]->(o)
                RETURN o.name as object, r as relationship, id(r) as rel_id
            """, {"subject": rel["subject"]})
            
            for existing_rel in existing:
                if existing_rel["object"] != rel["object"]:
                    contradictions.append({
                        "existing_fact_id": existing_rel["rel_id"],
                        "existing_statement": f"{rel['subject']} {rel['predicate']} {existing_rel['object']}",
                        "new_statement": f"{rel['subject']} {rel['predicate']} {rel['object']}",
                        "resolution_needed": True
                    })
        
        return contradictions
```

### 4.4 Contradiction Resolution Strategy

```python
# observer/contradiction_resolver.py

class ContradictionResolver:
    """
    Handles conflicting facts in the knowledge graph.
    
    Strategy:
    1. Temporal override: Newer facts supersede older ones
    2. Context preservation: Old facts marked as historical, not deleted
    3. Explicit tracking: Supersession chain maintained for auditing
    """
    
    async def resolve(self, contradiction: dict, graph_store) -> str:
        """
        Returns resolution action taken.
        """
        existing_id = contradiction["existing_fact_id"]
        new_statement = contradiction["new_statement"]
        
        # Mark existing fact as superseded (don't delete)
        await graph_store.query("""
            MATCH ()-[r]->() WHERE id(r) = $rel_id
            SET r.still_valid = false,
                r.superseded_at = datetime(),
                r.superseded_by = $new_statement
        """, {"rel_id": existing_id, "new_statement": new_statement})
        
        return f"Superseded: {contradiction['existing_statement']} -> {new_statement}"
```

---

## 5. ORCHESTRATION (LANGGRAPH)

### 5.1 State Machine Definition

```python
# orchestration/graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class ConversationState(TypedDict):
    # Input
    user_input: str
    conversation_history: list[dict]
    conversation_id: str
    
    # Retrieval
    retrieved_context: str
    retrieval_sources: list[str]
    
    # Generation
    assistant_response: str
    
    # Observer (async, doesn't block response)
    observer_triggered: bool
    observer_output: dict | None

def create_conversation_graph():
    """
    Main conversation flow:
    
    User Input -> Context Assembly -> LLM Generation -> Response
                                                    |
                                                    +-> Observer (async)
    """
    
    graph = StateGraph(ConversationState)
    
    # Nodes
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("trigger_observer", trigger_observer_node)
    
    # Edges
    graph.add_edge("assemble_context", "generate_response")
    graph.add_edge("generate_response", "trigger_observer")
    graph.add_edge("trigger_observer", END)
    
    # Entry point
    graph.set_entry_point("assemble_context")
    
    return graph.compile()

async def assemble_context_node(state: ConversationState) -> ConversationState:
    """Retrieve relevant context from memory stores."""
    assembler = get_context_assembler()  # Singleton
    
    context = await assembler.assemble(
        query=state["user_input"],
        conversation_history=state["conversation_history"]
    )
    
    return {
        **state,
        "retrieved_context": context
    }

async def generate_response_node(state: ConversationState) -> ConversationState:
    """Generate response using main LLM."""
    llm = get_main_llm()  # Ollama client for Qwen 3 14B
    
    system_prompt = build_system_prompt(state["retrieved_context"])
    
    response = await llm.generate(
        model="qwen3:14b",
        system=system_prompt,
        prompt=state["user_input"]
    )
    
    return {
        **state,
        "assistant_response": response
    }

async def trigger_observer_node(state: ConversationState) -> ConversationState:
    """Trigger async observer processing (non-blocking)."""
    observer = get_observer()  # Singleton
    
    # Fire and forget - don't await
    asyncio.create_task(
        observer.process_turn(
            user_message=state["user_input"],
            assistant_response=state["assistant_response"],
            conversation_id=state["conversation_id"],
            turn_index=len(state["conversation_history"])
        )
    )
    
    return {
        **state,
        "observer_triggered": True
    }
```

### 5.2 System Prompt Template

```python
# orchestration/prompts.py

SYSTEM_PROMPT_TEMPLATE = """You are a personal AI assistant with memory of past conversations. You know the user well and respond naturally, like a trusted friend who remembers everything.

## Your Personality
- Warm but not sycophantic
- Direct and honest
- Remembers context without being creepy about it
- Asks clarifying questions when needed
- Admits when you don't know something

## Context from Memory
{retrieved_context}

## Instructions
- Use the memory context naturally in your responses
- Don't explicitly say "according to my memory" or "you told me before"
- Just incorporate the knowledge as a friend would
- If the memory context seems outdated, ask if it's still accurate
- If you have no relevant memories, just respond naturally without them

## Current Conversation
Respond to the user's latest message below."""
```

---

## 6. VOICE INTERFACE

### 6.1 Audio Pipeline

```python
# voice/pipeline.py

import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
import numpy as np
import asyncio

class VoiceInterface:
    """
    Complete voice I/O pipeline.
    Uses wake word detection, streaming STT, and TTS.
    """
    
    def __init__(
        self,
        whisper_model: str = "medium",      # "tiny", "base", "small", "medium"
        piper_voice: str = "en_US-lessac-medium",
        sample_rate: int = 16000,
        wake_word: str = "hey assistant"     # Customizable
    ):
        self.stt = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        self.tts = PiperVoice.load(piper_voice)
        self.sample_rate = sample_rate
        self.wake_word = wake_word.lower()
        self.is_listening = False
    
    async def listen(self, timeout: float = 10.0) -> str | None:
        """
        Record audio until silence detected or timeout.
        Returns transcribed text.
        """
        audio_chunks = []
        silence_threshold = 0.01
        silence_duration = 0
        max_silence = 1.5  # seconds of silence to stop
        
        def callback(indata, frames, time, status):
            audio_chunks.append(indata.copy())
            
            # Simple VAD: check if audio level is below threshold
            level = np.abs(indata).mean()
            nonlocal silence_duration
            if level < silence_threshold:
                silence_duration += frames / self.sample_rate
            else:
                silence_duration = 0
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(0.1)
                
                if silence_duration > max_silence and len(audio_chunks) > 10:
                    break
                if asyncio.get_event_loop().time() - start_time > timeout:
                    break
        
        if not audio_chunks:
            return None
        
        audio = np.concatenate(audio_chunks)
        segments, _ = self.stt.transcribe(audio, language="en")
        return " ".join(segment.text for segment in segments)
    
    async def speak(self, text: str):
        """Convert text to speech and play."""
        # Piper generates audio as numpy array
        audio = self.tts.synthesize(text)
        sd.play(audio, self.sample_rate)
        sd.wait()
    
    async def run_voice_loop(self, conversation_handler):
        """
        Main voice interaction loop.
        Listens -> Processes -> Speaks -> Repeat
        """
        print("Voice interface active. Say 'hey assistant' to begin.")
        
        while True:
            # Listen for wake word (simplified - production would use Porcupine or similar)
            text = await self.listen(timeout=30.0)
            
            if text and self.wake_word in text.lower():
                await self.speak("Yes?")
                
                # Listen for actual query
                query = await self.listen(timeout=15.0)
                
                if query:
                    # Process through conversation handler
                    response = await conversation_handler(query)
                    await self.speak(response)
```

---

## 7. CONFIGURATION

### 7.1 Environment Variables

```bash
# .env

# LLM Settings
OLLAMA_HOST=http://localhost:11434
MAIN_MODEL=qwen3:14b
OBSERVER_MODEL=qwen3:1.7b
EMBEDDING_MODEL=nomic-embed-text:v1.5

# Database Paths
LANCEDB_PATH=./data/lancedb
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
REDIS_HOST=localhost
REDIS_PORT=6380

# Memory Settings
MAX_CONTEXT_TOKENS=3000
SLIDING_WINDOW_TOKENS=2000
TEMPORAL_DECAY_DAYS=30
VECTOR_SEARCH_TOP_K=15
GRAPH_SEARCH_TOP_K=10
RERANK_TOP_K=5

# Chunking Settings
SIMILARITY_THRESHOLD=0.85
MIN_CHUNK_SENTENCES=3
MAX_CHUNK_TOKENS=512

# Voice Settings
WHISPER_MODEL=medium
PIPER_VOICE=en_US-lessac-medium
WAKE_WORD=hey assistant

# Logging
LOG_LEVEL=INFO
```

### 7.2 Pydantic Settings

```python
# config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    ollama_host: str = "http://localhost:11434"
    main_model: str = "qwen3:14b"
    observer_model: str = "qwen3:1.7b"
    embedding_model: str = "nomic-embed-text:v1.5"
    
    # Databases
    lancedb_path: str = "./data/lancedb"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    redis_host: str = "localhost"
    redis_port: int = 6380
    
    # Memory
    max_context_tokens: int = 3000
    sliding_window_tokens: int = 2000
    temporal_decay_days: int = 30
    vector_search_top_k: int = 15
    graph_search_top_k: int = 10
    rerank_top_k: int = 5
    
    # Chunking
    similarity_threshold: float = 0.85
    min_chunk_sentences: int = 3
    max_chunk_tokens: int = 512
    
    # Voice
    whisper_model: str = "medium"
    piper_voice: str = "en_US-lessac-medium"
    wake_word: str = "hey assistant"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 8. IMPLEMENTATION PHASES

### Phase 1: Foundation (Week 1-2)
**Goal**: Basic chat loop with vector memory

- [ ] Set up project structure and dependencies
- [ ] Implement Ollama client wrapper
- [ ] Implement LanceDB vector store with schema
- [ ] Implement semantic chunker
- [ ] Build basic context assembly (vector only, no graph)
- [ ] Create simple CLI chat interface
- [ ] Test: Multi-turn conversation with memory recall

**Deliverable**: Can have conversations that remember facts from earlier in the same session.

### Phase 2: Knowledge Graph (Week 3-4)
**Goal**: Entity-relationship tracking

- [ ] Deploy FalkorDB via Docker
- [ ] Implement graph store client
- [ ] Implement Observer entity/relationship extraction
- [ ] Implement graph queries in context assembly
- [ ] Implement contradiction detection
- [ ] Test: Ask about entities mentioned in past conversations

**Deliverable**: Can answer "who is Sarah?" or "what phone do I have?" from graph.

### Phase 3: Observer Integration (Week 5-6)
**Goal**: Full async memory management

- [ ] Implement Observer with all extraction tasks
- [ ] Implement utility grading
- [ ] Implement pre-generated retrieval queries
- [ ] Add temporal decay to retrieval
- [ ] Implement cross-encoder reranking
- [ ] Test: Long-term memory coherence over many conversations

**Deliverable**: System autonomously builds and maintains memory without explicit commands.

### Phase 4: Voice Interface (Week 7-8)
**Goal**: Hands-free interaction

- [ ] Integrate Whisper.cpp for STT
- [ ] Integrate Piper for TTS
- [ ] Implement voice activity detection
- [ ] Implement wake word detection
- [ ] Build voice interaction loop
- [ ] Test: Full voice conversation with memory

**Deliverable**: Speak to the assistant and receive spoken responses.

### Phase 5: Polish & Optimization (Week 9-10)
**Goal**: Production-ready system

- [ ] Performance profiling and optimization
- [ ] Memory leak testing
- [ ] Edge case handling
- [ ] Conversation log management (archival, cleanup)
- [ ] Backup/restore functionality
- [ ] Documentation and user guide

**Deliverable**: Stable, performant system for daily use.

---

## 9. TESTING STRATEGY

### 9.1 Unit Tests
```python
# tests/test_chunker.py
def test_semantic_chunker_respects_min_sentences():
    """Chunks should never be smaller than min_chunk_sentences."""
    pass

# tests/test_observer.py
def test_utility_grading_discards_thanks():
    """'Thanks!' should be graded as DISCARD."""
    pass

def test_entity_extraction_finds_relationships():
    """'I work at Acme Corp' should extract WORKS_AT relationship."""
    pass

# tests/test_context_assembler.py
def test_temporal_decay_reduces_old_memories():
    """30-day-old memories should have ~50% score reduction."""
    pass
```

### 9.2 Integration Tests
```python
# tests/test_integration.py

async def test_memory_persists_across_sessions():
    """Information shared in session 1 should be retrievable in session 2."""
    
    # Session 1
    response1 = await chat("My cat's name is Whiskers")
    
    # Clear context, simulate new session
    clear_sliding_window()
    
    # Session 2
    response2 = await chat("What's my cat's name?")
    assert "Whiskers" in response2

async def test_contradiction_resolution():
    """Newer facts should supersede older ones."""
    
    await chat("I work at Company A")
    await chat("I started a new job at Company B last week")
    
    response = await chat("Where do I work?")
    assert "Company B" in response
    assert "Company A" not in response
```

### 9.3 Memory Coherence Tests
```python
# tests/test_memory_coherence.py

async def test_no_hallucinated_memories():
    """System should not claim to remember things never discussed."""
    
    # Fresh system with no history
    response = await chat("What's my favorite color?")
    assert "don't know" in response.lower() or "haven't told me" in response.lower()

async def test_memory_staleness_awareness():
    """System should handle potentially outdated information gracefully."""
    
    # Set up old memory
    inject_memory("User lives in Seattle", age_days=365)
    
    response = await chat("I'm looking for a good restaurant nearby")
    # Should either use the memory with caveat or ask for current location
    assert "Seattle" in response or "where" in response.lower()
```

---

## 10. COMMON ISSUES & SOLUTIONS

### Issue: VRAM Exhaustion
**Symptom**: OOM errors when running main model + reranker  
**Solution**: 
- Ensure reranker loads with `device_map="auto"` 
- Use `torch.cuda.empty_cache()` after reranking
- Consider CPU-only reranker if issues persist

### Issue: Slow Observer Processing
**Symptom**: Memory updates lagging behind conversation  
**Solution**:
- Observer runs on CPU by default - this is intentional
- Use Redis queue for batched processing during idle time
- Reduce extraction prompt complexity if >5s per turn

### Issue: Graph Query Timeouts
**Symptom**: FalkorDB queries taking >1s  
**Solution**:
- Add indexes: `CREATE INDEX FOR (p:Person) ON (p.name)`
- Limit traversal depth to 2 hops
- Use `LIMIT` clauses on all queries

### Issue: Poor Retrieval Quality
**Symptom**: System retrieves irrelevant memories  
**Solution**:
- Increase rerank_top_k and rely more on cross-encoder
- Tune similarity_threshold higher (0.88-0.92)
- Verify embedding model is properly loaded

### Issue: Voice Recognition Errors
**Symptom**: Whisper misinterprets speech frequently  
**Solution**:
- Upgrade to "medium" or "large" model
- Add domain-specific vocabulary to prompts
- Implement confirmation loop for ambiguous inputs

---

## 11. FUTURE ENHANCEMENTS

### 11.1 Potential Additions (Out of Scope for v1)
- **Multi-user support**: Separate memory stores per user
- **Web UI**: Browser-based interface alongside voice
- **Mobile app**: React Native client connecting to local server
- **Plugin system**: Add skills (calendar, email, smart home)
- **Fine-tuning**: Personalize base model on user's writing style
- **Emotional modeling**: Track and respond to user's emotional patterns

### 11.2 Model Upgrade Path
As local hardware improves or models become more efficient:
- Main LLM: Qwen 3 14B → TBD
- Observer: Qwen 3 4B → TBD
- Embedding: nomic-embed-text → BGE-M3 (multilingual support)

---

## 12. REFERENCES

- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Semantic Chunking Research](https://arxiv.org/abs/2312.06648)

---

*Last Updated: 2025*  
*Version: 1.0*  
*Author: Claude (Anthropic) in collaboration with User*
