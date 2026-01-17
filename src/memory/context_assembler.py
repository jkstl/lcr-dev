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
        reranker=None,  # Phase 4: Cross-encoder reranker
        max_context_tokens: int | None = None,
        sliding_window_turns: int | None = None,
        vector_search_top_k: int | None = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker
        self.max_context_tokens = max_context_tokens or settings.max_context_tokens
        self.sliding_window_turns = sliding_window_turns or settings.sliding_window_turns
        self.vector_search_top_k = vector_search_top_k or settings.vector_search_top_k
        
        # Use reranker settings if reranker is available
        if self.reranker:
            self.vector_candidates_k = settings.vector_candidates_k
            self.reranker_top_k = settings.reranker_top_k
    
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
        4. Previous assistant actions (if available)
        """
        # Step 1: Get sliding window (most recent turns)
        sliding_context = self._get_sliding_window(conversation_history)

        # Step 2: Search vector store for relevant memories
        # If reranker available, get more candidates for reranking
        search_k = self.vector_candidates_k if self.reranker else self.vector_search_top_k
        vector_results = await self.vector_store.search(
            query=query,
            top_k=search_k,
        )

        # Step 2.5: Rerank results if reranker is available (Phase 4)
        if self.reranker and vector_results:
            vector_results = self._rerank_results(query, vector_results)

        # Step 3: Query graph for entity facts (if graph store available)
        graph_facts = ""
        assistant_context = ""
        if self.graph_store:
            graph_facts = self._query_graph_facts(query)
            assistant_context = self._query_assistant_context(query)

        # Step 4: Format memories
        memory_context = self._format_memories(vector_results)

        # Step 5: Combine
        return self._build_final_context(sliding_context, memory_context, graph_facts, assistant_context)
    
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
        """Format vector search results with temporal weighting."""
        if not results:
            return ""
        
        from datetime import datetime
        
        # Add recency scores
        scored_results = []
        now = datetime.now()
        
        for r in results:
            # Get creation time
            created_str = r.get("created_at")
            if isinstance(created_str, str):
                try:
                    created = datetime.fromisoformat(created_str)
                    days_old = (now - created).days
                    # Exponential decay: recent=1.0, 30days=0.5, 90days=0.25
                    recency_score = 2 ** (-days_old / 30)
                except:
                    recency_score = 0.5  # Default if no date
            else:
                recency_score = 0.5
            
            scored_results.append((r, recency_score))
        
        # Sort by recency (most recent first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        lines = []
        for i, (r, score) in enumerate(scored_results, 1):
            summary = r.get("summary", "")
            content = r.get("content", "")
            text = summary if summary else content[:200]
            
            # Add temporal indicator for very recent (< 7 days)
            created_str = r.get("created_at", "")
            if isinstance(created_str, str):
                try:
                    created = datetime.fromisoformat(created_str)
                    days_old = (now - created).days
                    if days_old < 7:
                        text = f"[Recent] {text}"
                except:
                    pass
            
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

    def _query_assistant_context(self, query: str) -> str:
        """Query past assistant actions relevant to the current query."""
        if not self.graph_store:
            return ""

        # Extract potential topics from query
        words = query.lower().split()
        # Common topic keywords to look for
        topic_keywords = [w.strip(',.?!') for w in words if len(w) > 3]

        actions_lines = []
        seen_targets = set()

        # First, get recent assistant actions without topic filter
        all_actions = self.graph_store.query_assistant_actions(limit=5)

        for action in all_actions:
            target = action.get("target", "")
            if target in seen_targets:
                continue
            seen_targets.add(target)

            action_type = action.get("action", "")
            metadata = action.get("metadata", {})

            # Format the action
            if action_type == "RECOMMENDED":
                reason = metadata.get("reason", "")
                if reason:
                    actions_lines.append(f"- RECOMMENDED: {target} (for {reason})")
                else:
                    actions_lines.append(f"- RECOMMENDED: {target}")
            elif action_type == "SUGGESTED":
                purpose = metadata.get("purpose", "")
                if purpose:
                    actions_lines.append(f"- SUGGESTED: {target} ({purpose})")
                else:
                    actions_lines.append(f"- SUGGESTED: {target}")
            elif action_type == "EXPLAINED":
                actions_lines.append(f"- EXPLAINED: {target}")
            elif action_type == "ASKED_ABOUT":
                actions_lines.append(f"- ASKED_ABOUT: {target}")
            elif action_type == "OFFERED":
                actions_lines.append(f"- OFFERED: {target}")
            else:
                actions_lines.append(f"- {action_type}: {target}")

        return "\n".join(actions_lines) if actions_lines else ""

    def _build_final_context(self, sliding: str, memories: str, graph_facts: str = "", assistant_context: str = "") -> str:
        """Combine sliding window, memories, graph facts, and assistant history with temporal priority."""
        parts = []

        # Priority 1: Graph facts (current truth)
        if graph_facts:
            parts.append("## Current Facts (Knowledge Graph)")
            parts.append(graph_facts)
            parts.append("")

        # Priority 2: Previous Assistant Actions (avoid repetition)
        if assistant_context:
            parts.append("## Previous Assistant Actions")
            parts.append(assistant_context)
            parts.append("")

        # Priority 3: Recent memories (temporally weighted)
        if memories:
            parts.append("## Past Conversations (Recent First)")
            parts.append(memories)
            parts.append("")

        # Priority 4: Sliding window (immediate context)
        if sliding:
            parts.append("## Current Conversation")
            parts.append(sliding)

        return "\n".join(parts)
    
    def _rerank_results(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank vector search results using cross-encoder (Phase 4)."""
        if not results or not self.reranker:
            return results
        
        # Extract documents (use summary if available, else truncated content)
        documents = []
        for r in results:
            summary = r.get("summary", "")
            content = r.get("content", "")
            doc = summary if summary else content[:500]
            documents.append(doc)
        
        # Rerank using cross-encoder
        reranked_indices = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.reranker_top_k
        )
        
        # Return reordered results (only top_k)
        reranked_results = [results[idx] for idx, score in reranked_indices]
        
        return reranked_results
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars for English)."""
        return len(text) // 4
