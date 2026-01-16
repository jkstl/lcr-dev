"""Embedding generation using Ollama's embedding API."""

import httpx
import numpy as np
from src.config import settings


class Embedder:
    """Generate embeddings using Ollama's embedding model."""
    
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.embedding_model
        self._client = httpx.AsyncClient(base_url=self.host, timeout=60.0)
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        payload = {
            "model": self.model,
            "input": text,
        }
        
        response = await self._client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Ollama returns embeddings in "embeddings" array
        return data["embeddings"][0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        response = await self._client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()
        
        return data["embeddings"]
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
