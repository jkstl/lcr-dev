"""Context assembly from memory stores."""

from dataclasses import dataclass
from src.config import settings


@dataclass
class RetrievedContext:
    """A piece of retrieved context."""
    content: str
    summary: str
    source: str  # "vector" | "sliding_window"
    relevance_score: float


class ContextAssembler:
    """Assembles context for the LLM from memory and conversation history."""
    
    def __init__(
        self,
        vector_store,
        max_context_tokens: int | None = None,
        sliding_window_turns: int | None = None,
        vector_search_top_k: int | None = None,
    ):
        self.vector_store = vector_store
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
        """
        # Step 1: Get sliding window (most recent turns)
        sliding_context = self._get_sliding_window(conversation_history)
        
        # Step 2: Search vector store for relevant memories
        vector_results = await self.vector_store.search(
            query=query,
            top_k=self.vector_search_top_k,
        )
        
        # Step 3: Format memories
        memory_context = self._format_memories(vector_results)
        
        # Step 4: Combine
        return self._build_final_context(sliding_context, memory_context)
    
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
    
    def _build_final_context(self, sliding: str, memories: str) -> str:
        """Combine sliding window and memories into final context."""
        parts = []
        
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
