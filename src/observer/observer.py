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
    USER_EXTRACTION_PROMPT,
    ASSISTANT_EXTRACTION_PROMPT,
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
            model: Model to use for extraction (defaults to observer_model, 
                   a smaller model like qwen3:4b for faster processing)
            ollama_host: Ollama API host
            graph_store: Optional GraphStore instance for entity tracking
        """
        self.model = model or settings.observer_model
        self.host = ollama_host or settings.ollama_host
        self._llm: OllamaClient | None = None
        self.graph_store = graph_store
    
    async def _get_llm(self) -> OllamaClient:
        """Lazy initialize LLM client."""
        if self._llm is None:
            # Use num_gpu=0 to run Observer on CPU, freeing VRAM for main LLM
            self._llm = OllamaClient(host=self.host, model=self.model, num_gpu=0)
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

        # Run tasks sequentially to prevent CPU overload/Ollama hangs
        utility_grade = await self._grade_utility(combined)

        # Early return for discardable content
        if utility_grade == UtilityGrade.DISCARD:
            return ObserverOutput(
                utility_grade=utility_grade,
                utility_score=0.0,
                summary=None,
            )

        summary = await self._generate_summary(combined)

        # Dual extraction: user facts and assistant actions
        user_data = await self._extract_user_facts(user_message, assistant_response)

        # Skip assistant extraction for LOW utility (optimization)
        if utility_grade == UtilityGrade.LOW:
            assistant_data = {"entities": [], "relationships": []}
        else:
            assistant_data = await self._extract_assistant_actions(assistant_response, user_message)

        queries = await self._generate_retrieval_queries(combined)

        # Store in graph if available
        if self.graph_store:
            if user_data:
                await self._store_in_graph(user_data, source="USER")
            if assistant_data and assistant_data.get("relationships"):
                await self._store_in_graph(assistant_data, source="ASSISTANT")

        # Combine entities and relationships for output
        all_entities = user_data.get("entities", []) + assistant_data.get("entities", [])
        all_relationships = user_data.get("relationships", []) + assistant_data.get("relationships", [])

        return ObserverOutput(
            utility_grade=utility_grade,
            utility_score=self.UTILITY_SCORES[utility_grade],
            summary=summary,
            entities=all_entities,
            relationships=all_relationships,
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
            import traceback
            print(f"[Observer] Utility grading error: {type(e).__name__}: {e}")
            traceback.print_exc()
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
            import traceback
            print(f"[Observer] Summary generation error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return ""
    
    async def _extract_user_facts(self, user_text: str, assistant_text: str) -> dict:
        """Extract entities and relationships from what the user said."""
        llm = await self._get_llm()

        try:
            response = await llm.generate(
                prompt=USER_EXTRACTION_PROMPT.format(
                    user_text=user_text,
                    assistant_text=assistant_text
                ),
                model=self.model,
            )
            return self._parse_json(response)
        except Exception as e:
            import traceback
            print(f"[Observer] User fact extraction error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return {"entities": [], "relationships": []}

    async def _extract_assistant_actions(self, assistant_text: str, user_text: str) -> dict:
        """Extract what actions the assistant performed (recommendations, explanations, etc.)."""
        llm = await self._get_llm()

        try:
            response = await llm.generate(
                prompt=ASSISTANT_EXTRACTION_PROMPT.format(
                    assistant_text=assistant_text,
                    user_text=user_text
                ),
                model=self.model,
            )
            return self._parse_json(response)
        except Exception as e:
            import traceback
            print(f"[Observer] Assistant action extraction error: {type(e).__name__}: {e}")
            traceback.print_exc()
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
            import traceback
            print(f"[Observer] Query generation error: {type(e).__name__}: {e}")
            traceback.print_exc()
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
            data = json.loads(text)
            return self._normalize_extraction(data)
        except json.JSONDecodeError:
            # Return empty structure on parse failure
            if "[" in text:
                return []
            return {"entities": [], "relationships": []}

    def _normalize_extraction(self, data: dict | list) -> dict | list:
        """Normalize extraction data to handle schema variants from LLM."""
        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return {"entities": [], "relationships": []}

        # Normalize entities - convert dict format to list format
        entities = data.get("entities", [])
        if isinstance(entities, dict):
            # Convert {"Mom": "Person", "Justine": "Person"} to [{"name": "Mom", "type": "Person"}, ...]
            normalized_entities = []
            for name, entity_type in entities.items():
                if isinstance(entity_type, str):
                    normalized_entities.append({"name": name, "type": entity_type, "attributes": {}})
                elif isinstance(entity_type, dict):
                    normalized_entities.append({"name": name, **entity_type})
            entities = normalized_entities

        # Normalize relationships
        relationships = data.get("relationships", [])
        if isinstance(relationships, dict):
            # Convert weird dict format to list format
            # e.g., {"SIBLING_OF": ["User", "Justine"]} -> [{"subject": "User", "predicate": "SIBLING_OF", "object": "Justine"}]
            normalized_rels = []
            for predicate, value in relationships.items():
                if isinstance(value, list) and len(value) >= 2:
                    normalized_rels.append({
                        "subject": value[0],
                        "predicate": predicate,
                        "object": value[1],
                        "metadata": {}
                    })
            relationships = normalized_rels
        elif isinstance(relationships, list):
            # Normalize each relationship - convert "relation" to "predicate"
            normalized_rels = []
            for rel in relationships:
                if isinstance(rel, dict):
                    normalized_rel = dict(rel)
                    # Convert "relation" key to "predicate"
                    if "relation" in normalized_rel and "predicate" not in normalized_rel:
                        normalized_rel["predicate"] = normalized_rel.pop("relation")
                    # Ensure metadata exists
                    if "metadata" not in normalized_rel:
                        normalized_rel["metadata"] = {}
                    normalized_rels.append(normalized_rel)
            relationships = normalized_rels

        return {"entities": entities, "relationships": relationships}
    
    async def _store_in_graph(self, entities_data: dict, source: str = "USER"):
        """Store extracted entities and relationships in the graph.

        Args:
            entities_data: Dict with 'entities' and 'relationships' lists
            source: Attribution source - "USER" or "ASSISTANT"
        """
        if not self.graph_store:
            return

        print(f"\n[Observer DEBUG] Entities data (source={source}): {entities_data}")

        try:
            # Ensure ASSISTANT node exists if storing assistant actions
            if source == "ASSISTANT":
                self.graph_store.get_or_create_assistant_node()

            # Store entities
            for entity in entities_data.get("entities", []):
                entity_type = entity.get("type", "Concept")
                entity_name = entity.get("name")

                # Skip the ASSISTANT entity - it's created separately
                if entity_name == "ASSISTANT":
                    continue

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

                print(f"[Observer DEBUG] Processing relationship: {subject} -{predicate}-> {obj} (source={source})")

                if not (subject and predicate and obj):
                    print(f"[Observer DEBUG] Skipping incomplete relationship")
                    continue

                # Check for contradictions (only for USER facts, not assistant actions)
                if source == "USER":
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

                # Add new relationship with source attribution
                success = self.graph_store.add_relationship(
                    subject=subject,
                    predicate=predicate,
                    object_name=obj,
                    metadata=rel.get("metadata", {}),
                    source=source
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
