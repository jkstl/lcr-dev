"""Observer system for async memory extraction and grading.

The Observer runs after each conversation turn to:
1. Grade the utility of the conversation (worth remembering?)
2. Extract entities and relationships
3. Generate a concise summary
4. Pre-generate retrieval queries for better future search
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.models.llm import OllamaClient
from src.config import settings
from src.observer.prompts import (
    UTILITY_GRADING_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
    RETRIEVAL_QUERIES_PROMPT,
)


class UtilityGrade(Enum):
    """Memory utility grades."""
    DISCARD = "discard"   # Greetings, thanks, small talk
    LOW = "low"           # General discussion, no new facts
    MEDIUM = "medium"     # Preferences, opinions, feelings
    HIGH = "high"         # Facts, schedules, relationships


@dataclass
class ObserverOutput:
    """Output from the Observer processing a conversation turn."""
    utility_grade: UtilityGrade
    utility_score: float
    summary: str | None
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)


class Observer:
    """
    Async post-generation reflection system.
    Runs after every assistant response to extract and grade memories.
    """
    
    # Map utility grades to numeric scores
    UTILITY_SCORES = {
        UtilityGrade.DISCARD: 0.0,
        UtilityGrade.LOW: 0.3,
        UtilityGrade.MEDIUM: 0.6,
        UtilityGrade.HIGH: 1.0,
    }
    
    def __init__(
        self,
        model: str | None = None,
        ollama_host: str | None = None,
        graph_store=None,  # Optional graph store
    ):
        """
        Initialize the Observer.
        
        Args:
            model: Model to use for extraction (defaults to main model for now,
                   ideally would be a smaller model like phi3.5)
            ollama_host: Ollama API host
            graph_store: Optional GraphStore instance for entity tracking
        """
        self.model = model or settings.main_model
        self.host = ollama_host or settings.ollama_host
        self._llm: OllamaClient | None = None
        self.graph_store = graph_store
    
    async def _get_llm(self) -> OllamaClient:
        """Lazy initialize LLM client."""
        if self._llm is None:
            self._llm = OllamaClient(host=self.host, model=self.model)
        return self._llm
    
    async def process_turn(
        self,
        user_message: str,
        assistant_response: str,
    ) -> ObserverOutput:
        """
        Process a conversation turn and extract memory-relevant information.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            
        Returns:
            ObserverOutput with utility grade, summary, entities, relationships, and queries
        """
        combined = f"USER: {user_message}\nASSISTANT: {assistant_response}"
        
        # Run all extraction tasks in parallel for efficiency
        utility_task = self._grade_utility(combined)
        summary_task = self._generate_summary(combined)
        entity_task = self._extract_entities(combined)
        query_task = self._generate_retrieval_queries(combined)
        
        utility_grade, summary, entities_data, queries = await asyncio.gather(
            utility_task, summary_task, entity_task, query_task
        )
        
        # Early return for discardable content
        if utility_grade == UtilityGrade.DISCARD:
            return ObserverOutput(
                utility_grade=utility_grade,
                utility_score=0.0,
                summary=None,
            )
        
        # Store entities and relationships in graph if available
        if self.graph_store and entities_data:
            await self._store_in_graph(entities_data)
        
        return ObserverOutput(
            utility_grade=utility_grade,
            utility_score=self.UTILITY_SCORES[utility_grade],
            summary=summary,
            entities=entities_data.get("entities", []),
            relationships=entities_data.get("relationships", []),
            retrieval_queries=queries,
        )
    
    async def _grade_utility(self, text: str) -> UtilityGrade:
        """Grade how worth remembering this conversation turn is."""
        llm = await self._get_llm()
        
        try:
            response = await llm.generate(
                prompt=UTILITY_GRADING_PROMPT.format(text=text),
                model=self.model,
            )
            
            # Parse the grade from the response
            grade_str = response.strip().upper()
            
            # Handle potential thinking wrapper from Qwen3
            if "DISCARD" in grade_str:
                return UtilityGrade.DISCARD
            elif "HIGH" in grade_str:
                return UtilityGrade.HIGH
            elif "MEDIUM" in grade_str:
                return UtilityGrade.MEDIUM
            elif "LOW" in grade_str:
                return UtilityGrade.LOW
            else:
                # Default to MEDIUM if unclear
                return UtilityGrade.MEDIUM
                
        except Exception as e:
            # On error, default to MEDIUM to avoid losing data
            print(f"[Observer] Utility grading error: {e}")
            return UtilityGrade.MEDIUM
    
    async def _generate_summary(self, text: str) -> str:
        """Generate a concise summary of the conversation turn."""
        llm = await self._get_llm()
        
        try:
            response = await llm.generate(
                prompt=SUMMARY_PROMPT.format(text=text),
                model=self.model,
            )
            return response.strip()
        except Exception as e:
            print(f"[Observer] Summary generation error: {e}")
            return ""
    
    async def _extract_entities(self, text: str) -> dict:
        """Extract entities and relationships from the conversation."""
        llm = await self._get_llm()
        
        try:
            response = await llm.generate(
                prompt=ENTITY_EXTRACTION_PROMPT.format(text=text),
                model=self.model,
            )
            return self._parse_json(response)
        except Exception as e:
            print(f"[Observer] Entity extraction error: {e}")
            return {"entities": [], "relationships": []}
    
    async def _generate_retrieval_queries(self, text: str) -> list[str]:
        """Generate questions that this memory could answer."""
        llm = await self._get_llm()
        
        try:
            response = await llm.generate(
                prompt=RETRIEVAL_QUERIES_PROMPT.format(text=text),
                model=self.model,
            )
            queries = self._parse_json(response)
            if isinstance(queries, list):
                return queries
            return []
        except Exception as e:
            print(f"[Observer] Query generation error: {e}")
            return []
    
    def _parse_json(self, text: str) -> dict | list:
        """Parse JSON from LLM response, handling common issues."""
        # Try to find JSON in the response
        text = text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Try to find JSON object or array
        start_obj = text.find("{")
        start_arr = text.find("[")
        
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            # Find matching closing brace
            end = text.rfind("}")
            if end != -1:
                text = text[start_obj:end + 1]
        elif start_arr != -1:
            # Find matching closing bracket
            end = text.rfind("]")
            if end != -1:
                text = text[start_arr:end + 1]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return empty structure on parse failure
            if "[" in text:
                return []
            return {"entities": [], "relationships": []}
    
    async def _store_in_graph(self, entities_data: dict):
        """Store extracted entities and relationships in the graph."""
        if not self.graph_store:
            return
        
        print(f"\n[Observer DEBUG] Entities data: {entities_data}")
        
        try:
            # Store entities
            for entity in entities_data.get("entities", []):
                entity_type = entity.get("type", "Concept")
                entity_name = entity.get("name")
                
                print(f"[Observer DEBUG] Processing entity: {entity_name} (type: {entity_type})")
                
                if entity_type == "Person":
                    self.graph_store.add_person(
                        name=entity["name"],
                        relationship_to_user=entity.get("attributes", {}).get("relationship", "unknown"),
                        attributes=entity.get("attributes")
                    )
                    print(f"[Observer DEBUG] Added Person: {entity_name}")
                else:
                    # Map entity types to categories
                    category_map = {
                        "Technology": "technology",
                        "Place": "place",
                        "Organization": "organization",
                        "Event": "event",
                        "Concept": "concept"
                    }
                    self.graph_store.add_entity(
                        name=entity["name"],
                        category=category_map.get(entity_type, "concept"),
                        attributes=entity.get("attributes")
                    )
                    print(f"[Observer DEBUG] Added Entity: {entity_name} ({category_map.get(entity_type, 'concept')})")
            
            # Store relationships with contradiction detection
            for rel in entities_data.get("relationships", []):
                subject = rel.get("subject")
                predicate = rel.get("predicate")
                obj = rel.get("object")
                
                print(f"[Observer DEBUG] Processing relationship: {subject} -{predicate}-> {obj}")
                
                if not (subject and predicate and obj):
                    print(f"[Observer DEBUG] Skipping incomplete relationship")
                    continue
                
                # Check for contradictions
                contradictions = self.graph_store.check_contradictions(
                    subject=subject,
                    predicate=predicate,
                    new_object=obj
                )
                
                if contradictions:
                    print(f"[Observer DEBUG] Found contradictions: {contradictions}")
                
                # Supersede old relationships if contradictions found
                for contradiction in contradictions:
                    print(f"[Observer DEBUG] Superseding: {subject} -{predicate}-> {contradiction['existing_object']}")
                    self.graph_store.supersede_relationship(
                        subject=subject,
                        predicate=predicate,
                        old_object=contradiction["existing_object"]
                    )
                
                # Add new relationship
                success = self.graph_store.add_relationship(
                    subject=subject,
                    predicate=predicate,
                    object_name=obj,
                    metadata=rel.get("metadata", {})
                )
                print(f"[Observer DEBUG] Added relationship: {success}")
        
        except Exception as e:
            print(f"[Observer] Error storing in graph: {e}")
            import traceback
            traceback.print_exc()
    
    async def close(self):
        """Clean up resources."""
        if self._llm:
            await self._llm.close()
            self._llm = None
