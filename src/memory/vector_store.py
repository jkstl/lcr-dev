"""LanceDB vector store for memory persistence."""

import lancedb
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import uuid

from src.config import settings
from src.models.embedder import Embedder


class MemoryChunk(BaseModel):
    """Schema for vector store entries."""
    id: str
    content: str
    summary: str
    embedding: list[float]
    
    # Metadata
    chunk_type: str  # "conversation" | "document" | "fact"
    conversation_id: str
    turn_index: int
    
    # Temporal
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    
    # Observer-generated fields
    utility_score: float = 0.5  # 0.0-1.0 from Observer grading
    retrieval_queries: str = ""  # JSON-encoded list of pre-generated queries


class VectorStore:
    """LanceDB-backed vector store for memory."""
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.lancedb_path
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self._embedder: Embedder | None = None
        self._init_table()
    
    def _init_table(self):
        """Initialize the memories table if it doesn't exist."""
        if "memories" not in self.db.table_names():
            # Create with empty schema - will be populated on first insert
            self._table = None
        else:
            self._table = self.db.open_table("memories")
    
    async def _get_embedder(self) -> Embedder:
        """Lazy initialize embedder."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder
    
    async def add_memory(
        self,
        content: str,
        summary: str,
        conversation_id: str,
        turn_index: int,
        chunk_type: str = "conversation",
        utility_score: float = 0.5,
        retrieval_queries: list[str] | None = None,
    ) -> str:
        """Add a new memory to the store."""
        import json
        
        embedder = await self._get_embedder()
        embedding = await embedder.embed(content)
        
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Encode retrieval queries as JSON string for LanceDB
        queries_json = json.dumps(retrieval_queries or [])
        
        chunk = MemoryChunk(
            id=memory_id,
            content=content,
            summary=summary,
            embedding=embedding,
            chunk_type=chunk_type,
            conversation_id=conversation_id,
            turn_index=turn_index,
            created_at=now,
            last_accessed_at=now,
            access_count=0,
            utility_score=utility_score,
            retrieval_queries=queries_json,
        )
        
        data = [chunk.model_dump()]
        
        if self._table is None:
            self._table = self.db.create_table("memories", data)
        else:
            self._table.add(data)
        
        return memory_id
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar memories."""
        if self._table is None:
            return []
        
        embedder = await self._get_embedder()
        query_embedding = await embedder.embed(query)
        
        results = (
            self._table
            .search(query_embedding)
            .limit(top_k)
            .to_list()
        )
        
        # Update access counts (in background would be better)
        for r in results:
            r["access_count"] = r.get("access_count", 0) + 1
            r["last_accessed_at"] = datetime.now()
        
        return results
    
    def get_all_memories(self) -> list[dict]:
        """Get all memories (for debugging)."""
        if self._table is None:
            return []
        return self._table.to_pandas().to_dict(orient="records")
    
    def count(self) -> int:
        """Count total memories."""
        if self._table is None:
            return 0
        return len(self._table)
    
    async def close(self):
        """Clean up resources."""
        if self._embedder:
            await self._embedder.close()
            self._embedder = None
