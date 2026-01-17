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

RELATIONSHIP TYPES:

Ownership & Possession:
- OWNS, RECEIVED_FROM

Social & Professional:
- KNOWS, WORKS_AT, MARRIED_TO, COWORKER_OF, FRIENDS_WITH, FAN_OF

Location & Spatial:
- LIVES_IN, LOCATED_AT

Temporal Intentions (capture future plans/intentions):
- PLANS_TO (future intention to do something with entity)
- INTENDS_TO (stronger future commitment)
- CONSIDERING (evaluating future action)

Temporal States (capture current/past/future states):
- CURRENTLY_USING_AS (present purpose/use of entity)
- USED_TO (past purpose/use that no longer applies)
- IN_STATE (current condition of entity)
- WAS_IN_STATE (past condition of entity)

Preferences & Feelings:
- PREFERS, FEELS_ABOUT

Scheduling:
- SCHEDULED

For relationships not covered above (e.g., infrastructure, technical systems):
- Use RELATED_TO with relationship_type in metadata
- Example: {{"subject": "Server1", "predicate": "RELATED_TO", "object": "Router1", 
           "metadata": {{"relationship_type": "connected_to", "port": "eth0"}}}}
- Common technical types: connected_to, depends_on, routes_to, listens_on, proxies_to, backs_up_to

EXTRACTION RULES:
1. Extract ONLY what is explicitly stated - do not infer
2. For temporal changes (was/now/planning), create SEPARATE relationships for each state
3. Include temporal context in metadata when mentioned (timeframe, reason, purpose)
4. Do NOT use generic types as names (e.g., use "Mom" instead of "Person", "Philadelphia" instead of "Place")
5. Return empty arrays if nothing found

EXAMPLES:

Example 1 - Family members:
Input: "My mom and sister are visiting from Boston"
Output:
{{
  "entities": [
    {{"name": "Mom", "type": "Person", "attributes": {{"relationship": "mother"}}}},
    {{"name": "Sister", "type": "Person", "attributes": {{"relationship": "sister"}}}},
    {{"name": "Boston", "type": "Place", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "Mom", "predicate": "LOCATED_AT", "object": "Boston", "metadata": {{"timeframe": "from"}}}},
    {{"subject": "Sister", "predicate": "LOCATED_AT", "object": "Boston", "metadata": {{"timeframe": "from"}}}}
  ]
}}

Example 2 - Temporal intent change:
Input: "I was planning to sell my Dell laptop but now I'm using it as a home server"
Output:
{{
  "entities": [
    {{"name": "USER", "type": "Person", "attributes": {{}}}},
    {{"name": "Dell laptop", "type": "Technology", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "USER", "predicate": "OWNS", "object": "Dell laptop", "metadata": {{}}}},
    {{"subject": "USER", "predicate": "PLANS_TO", "object": "sell Dell laptop", "metadata": {{"action": "sell", "timeframe": "past intention"}}}},
    {{"subject": "USER", "predicate": "CURRENTLY_USING_AS", "object": "Dell laptop", "metadata": {{"purpose": "home server", "timeframe": "now"}}}}
  ]
}}

Example 2 - State transition:
Input: "My iPhone was broken but I just got it repaired"
Output:
{{
  "entities": [
    {{"name": "USER", "type": "Person", "attributes": {{}}}},
    {{"name": "iPhone", "type": "Technology", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "USER", "predicate": "OWNS", "object": "iPhone", "metadata": {{}}}},
    {{"subject": "iPhone", "predicate": "WAS_IN_STATE", "object": "broken", "metadata": {{"timeframe": "past"}}}},
    {{"subject": "iPhone", "predicate": "IN_STATE", "object": "repaired", "metadata": {{"timeframe": "current"}}}}
  ]
}}
"""


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
