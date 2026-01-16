"""LCR - Local Cognitive RAG System

Main entry point with CLI chat interface.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.config import settings, get_data_dir
from src.models.llm import OllamaClient
from src.memory.vector_store import VectorStore
from src.memory.graph_store import GraphStore
from src.memory.context_assembler import ContextAssembler
from src.observer.observer import Observer, UtilityGrade


# System prompt for the assistant
SYSTEM_PROMPT = """You are a personal AI assistant with memory of past conversations. You know the user well and respond naturally, like a trusted friend who remembers everything.

## Your Personality
- Warm but not sycophantic
- Direct and honest
- Remembers context without being creepy about it
- Asks clarifying questions when needed
- Admits when you don't know something

## Context from Memory
{context}

## Instructions
- Use the memory context naturally in your responses
- Don't explicitly say "according to my memory" or "you told me before"
- Just incorporate the knowledge as a friend would
- If you have no relevant memories, just respond naturally without them

Respond to the user's latest message."""


class LCRAssistant:
    """Main LCR assistant class."""
    
    def __init__(self):
        self.console = Console()
        self.conversation_id = str(uuid.uuid4())
        self.conversation_history: list[dict] = []
        self.turn_index = 0
        
        # Initialize components
        get_data_dir()  # Ensure directories exist
        self.vector_store = VectorStore()
        
        # Try to initialize graph store (optional - may not have Docker running)
        try:
            self.graph_store = GraphStore()
        except Exception as e:
            print(f"[Warning] Could not connect to FalkorDB: {e}")
            print("[Warning] Graph features disabled. Start FalkorDB with 'docker-compose up -d'")
            self.graph_store = None
        
        self.context_assembler = ContextAssembler(self.vector_store)
        self.observer = Observer(graph_store=self.graph_store)
        self.llm: OllamaClient | None = None
    
    async def _get_llm(self) -> OllamaClient:
        """Lazy initialize LLM client."""
        if self.llm is None:
            self.llm = OllamaClient()
        return self.llm
    
    async def chat_stream(self, user_input: str):
        """Process a user message and stream the response."""
        llm = await self._get_llm()
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Assemble context from memory
        context = await self.context_assembler.assemble(
            query=user_input,
            conversation_history=self.conversation_history,
        )
        
        # Build system prompt with context
        system = SYSTEM_PROMPT.format(context=context if context else "(No relevant memories yet)")
        
        # Generate response with streaming
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
        
        # Collect full response while yielding chunks
        full_response = ""
        async for chunk in llm.stream_chat(messages, system=system):
            full_response += chunk
            yield chunk
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Store the turn in memory (fire-and-forget)
        asyncio.create_task(self._store_memory(user_input, full_response))
        
        self.turn_index += 1
    
    async def chat(self, user_input: str) -> str:
        """Process a user message and return assistant response."""
        llm = await self._get_llm()
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Assemble context from memory
        context = await self.context_assembler.assemble(
            query=user_input,
            conversation_history=self.conversation_history,
        )
        
        # Build system prompt with context
        system = SYSTEM_PROMPT.format(context=context if context else "(No relevant memories yet)")
        
        # Generate response
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history
        ]
        
        response = await llm.chat(messages, system=system)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Store the turn in memory (fire-and-forget - don't block response)
        asyncio.create_task(self._store_memory(user_input, response))
        
        self.turn_index += 1
        return response
    
    async def _store_memory(self, user_input: str, assistant_response: str):
        """Store the conversation turn in memory using Observer."""
        # Use Observer to analyze the turn
        observer_output = await self.observer.process_turn(
            user_message=user_input,
            assistant_response=assistant_response,
        )
        
        # Skip storing if Observer says to discard
        if observer_output.utility_grade == UtilityGrade.DISCARD:
            return
        
        combined = f"User: {user_input}\nAssistant: {assistant_response}"
        
        await self.vector_store.add_memory(
            content=combined,
            summary=observer_output.summary or f"User shared: {user_input[:100]}",
            conversation_id=self.conversation_id,
            turn_index=self.turn_index,
            utility_score=observer_output.utility_score,
            retrieval_queries=observer_output.retrieval_queries,
        )
    
    def _save_conversation_log(self):
        """Save conversation to disk."""
        log_path = Path(settings.conversations_path) / f"{self.conversation_id}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "id": self.conversation_id,
            "started_at": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "ended_at": datetime.now().isoformat(),
            "messages": self.conversation_history,
            "total_turns": self.turn_index,
        }
        
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return log_path
    
    async def run_cli(self):
        """Run the interactive CLI interface."""
        self.console.print(Panel.fit(
            "[bold blue]LCR - Local Cognitive RAG System[/bold blue]\n"
            "[dim]A memory-enhanced AI assistant[/dim]\n\n"
            f"Model: {settings.main_model}\n"
            f"Memory count: {self.vector_store.count()}",
            title="Welcome"
        ))
        self.console.print("\n[dim]Type 'quit' or 'exit' to end the conversation.[/dim]\n")
        
        try:
            while True:
                # Get user input
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() in ("quit", "exit", "bye"):
                    break
                
                if not user_input.strip():
                    continue
                
                # Stream response
                self.console.print()
                self.console.print("[bold blue]Assistant[/bold blue]")
                
                response_text = ""
                async for chunk in self.chat_stream(user_input):
                    self.console.print(chunk, end="")
                    response_text += chunk
                
                self.console.print()  # Newline after response
                self.console.print()
        
        except KeyboardInterrupt:
            self.console.print("\n[dim]Interrupted.[/dim]")
        
        finally:
            # Save conversation
            log_path = self._save_conversation_log()
            self.console.print(f"\n[dim]Conversation saved to: {log_path}[/dim]")
            
            # Cleanup
            if self.llm:
                await self.llm.close()
            await self.vector_store.close()
            await self.observer.close()
            if self.graph_store:
                self.graph_store.close()
            
            self.console.print(Panel.fit(
                f"[dim]Memories stored: {self.vector_store.count()}[/dim]",
                title="Session ended"
            ))


async def main():
    """Main entry point."""
    assistant = LCRAssistant()
    await assistant.run_cli()


if __name__ == "__main__":
    asyncio.run(main())
