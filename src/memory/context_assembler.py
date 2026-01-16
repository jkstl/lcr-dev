"""Context assembly from memory stores."""

from dataclasses import dataclass
from src.config import settings


@dataclass
class RetrievedContext:
    """A piece of retrieved context."""
    content: str
    summary: str
    source: str  # "vector" | "graph" | "sliding_window"
    relevance_score: float


class ContextAssembler:
    """Assembles context for the LLM from memory and conversation history."""
    
    def __init__(
        self,
        vector_store,
        graph_store=None,
        max_context_tokens: int | None = None,
        sliding_window_turns: int | None = None,
        vector_search_top_k: int | None = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        self.sliding_window_turns = sliding_window_turns or settings.sliding_window_turns
        self.vector_search_top_k = vector_search_top_k or settings.vector_search_top_k
    
    async def assemble(
        self,
        query: str,
        conversation_history: list[dict],
    ) -> str:
        """
        Assemble context from:
        1. Sliding window of recent conversation
        2. Vector similarity search of past memories
        3. Graph entity facts (if available)
        """
        # Step 1: Get sliding window (most recent turns)
        sliding_context = self._get_sliding_window(conversation_history)
        
        # Step 2: Search vector store for relevant memories
        vector_results = await self.vector_store.search(
            query=query,
            top_k=self.vector_search_top_k,
        )
        
        # Step 3: Query graph for entity facts (if graph store available)
        graph_facts = ""
        if self.graph_store:
            graph_facts = self._query_graph_facts(query)
        
        # Step 4: Format memories
        memory_context = self._format_memories(vector_results)
        
        # Step 5: Combine
        return self._build_final_context(sliding_context, memory_context, graph_facts)
    
    def _get_sliding_window(self, history: list[dict]) -> str:
        """Get the most recent conversation turns."""
        if not history:
            return ""
        
        # Take last N turns
        recent = history[-self.sliding_window_turns:]
        
        lines = []
        for msg in recent:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _format_memories(self, results: list[dict]) -> str:
        """Format vector search results as context."""
        if not results:
            return ""
        
        lines = []
        for i, r in enumerate(results, 1):
            summary = r.get("summary", "")
            content = r.get("content", "")
            # Use summary if available, otherwise content preview
            text = summary if summary else content[:200]
            lines.append(f"[Memory {i}] {text}")
        
        return "\n".join(lines)
    
    def _query_graph_facts(self, query: str) -> str:
        """Query graph for facts about entities mentioned in query."""
        if not self.graph_store:
            return ""
        
        # Simple entity extraction from query (look for capitalized words)
        import re
        words = query.split()
        potential_entities = [w.strip(',.?!') for w in words if w[0].isupper() and len(w) > 1]
        
        if not potential_entities:
            return ""
        
        facts_lines = []
        seen_entities = set()
        
        for entity in potential_entities:
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            
            # Query graph for facts about this entity
            facts = self.graph_store.query_entity_facts(entity, limit=3)
            
            if facts:
                facts_lines.append(f"**{entity}:**")
                for fact in facts:
                    rel = fact.get("relationship", "")
                    related = fact.get("related_entity", "")
                    facts_lines.append(f"  - {rel}: {related}")
        
        return "\n".join(facts_lines) if facts_lines else ""
    
    def _build_final_context(self, sliding: str, memories: str, graph_facts: str = "") -> str:
        """Combine sliding window, memories, and graph facts into final context."""
        parts = []
        
        if graph_facts:
            parts.append("## Known Facts (from Knowledge Graph)")
            parts.append(graph_facts)
            parts.append("")
        
        if memories:
            parts.append("## Relevant Memories from Past Conversations")
            parts.append(memories)
            parts.append("")
        
        if sliding:
            parts.append("## Recent Conversation")
            parts.append(sliding)
        
        return "\n".join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars for English)."""
        return len(text) // 4
