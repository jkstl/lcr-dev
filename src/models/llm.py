"""Ollama LLM client wrapper for chat and generation."""

import httpx
from typing import AsyncGenerator
from src.config import settings


class OllamaClient:
    """Async client for Ollama API."""
    
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.main_model
        # Increased timeout for model swapping scenarios (main ↔ observer models)
        self._client = httpx.AsyncClient(base_url=self.host, timeout=300.0)
    
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a response from the LLM."""
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        
        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    
    async def chat(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
    ) -> str:
        """Chat with the LLM using message history."""
        model = model or self.model
        
        # Prepend system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    
    async def stream_chat(
        self,
        messages: list[dict],
        system: str | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat responses token by token."""
        model = model or self.model
        
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


# Convenience function for quick generation
async def generate(prompt: str, system: str | None = None) -> str:
    """Quick generation without managing client lifecycle."""
    async with OllamaClient() as client:
        return await client.generate(prompt, system=system)
