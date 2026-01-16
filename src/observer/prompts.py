"""Prompt templates for the Observer system."""

UTILITY_GRADING_PROMPT = """Rate the memory-worthiness of this conversation turn.

TURN:
{text}

Rules:
- DISCARD: Greetings, thanks, acknowledgments, small talk with no personal information
- LOW: General discussion but no concrete facts about the user
- MEDIUM: Contains user preferences, opinions, or feelings  
- HIGH: Contains facts (schedules, relationships, owned items, work info, names)

Respond with exactly one word: DISCARD, LOW, MEDIUM, or HIGH"""


ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this conversation.

TURN:
{text}

Output valid JSON only:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "relationship type", "object": "entity2", "metadata": {{}}}}
    ]
}}

Common relationship types: WORKS_AT, LIVES_IN, OWNS, KNOWS, FEELS_ABOUT, PREFERS, SCHEDULED, MARRIED_TO, COWORKER_OF, FRIENDS_WITH, FAN_OF, RECEIVED_FROM

Only extract what is explicitly stated. Do not infer.
If no entities or relationships found, return empty arrays."""


SUMMARY_PROMPT = """Summarize this conversation turn in exactly one sentence.
Focus on factual information shared by the user.

TURN:
{text}

ONE SENTENCE SUMMARY:"""


RETRIEVAL_QUERIES_PROMPT = """What questions could this conversation turn answer in the future?
Generate 2-3 natural questions a user might ask that this memory would help answer.

TURN:
{text}

Output as JSON array of strings:
["question 1", "question 2", "question 3"]"""
