"""Cross-encoder reranker for improved memory retrieval relevance.

Uses BGE-Reranker-v2-m3 to rerank vector search results with more accurate
semantic similarity scoring. Cross-encoders jointly encode query+document,
providing better relevance than bi-encoder embeddings alone.
"""

from sentence_transformers import CrossEncoder
import torch
from typing import Optional

from src.config import settings


class Reranker:
    """Cross-encoder reranker using BGE-Reranker-v2-m3.
    
    This reranks vector search candidates to select the most contextually
    relevant results for the query.
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-reranker-v2-m3)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for the model
        """
        self.model_name = model_name or settings.reranker_model
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.max_length = max_length
        
        print(f"[Reranker] Loading {self.model_name} on {self.device}...")
        
        try:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length
            )
            print(f"[Reranker] Loaded successfully on {self.device}")
        except Exception as e:
            print(f"[Reranker] Error loading model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (default: all documents)
            
        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        # Create query-document pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Create indexed scores
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            return indexed_scores[:top_k]
        
        return indexed_scores
    
    def get_vram_usage(self) -> str:
        """Get approximate VRAM usage information."""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "N/A (CPU mode)"
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'model'):
            del self.model
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
