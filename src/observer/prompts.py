"""Prompt templates for the Observer system."""

UTILITY_GRADING_PROMPT = """Rate the memory-worthiness of this conversation turn.

TURN:
{text}

Rules:
- DISCARD: Greetings, thanks, acknowledgments, small talk with no personal information
- LOW: General discussion but no concrete facts about the user
- MEDIUM: Contains user preferences, opinions, or feelings  
- HIGH: Contains facts (schedules, relationships, family, romantic partners, breakups, life events, owned items, work info, names)

Respond with exactly one word: DISCARD, LOW, MEDIUM, or HIGH"""


USER_EXTRACTION_PROMPT = """Extract entities and relationships from what the USER said.

USER MESSAGE:
{user_text}

ASSISTANT RESPONSE (context only - do NOT extract from this):
{assistant_text}

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

Family:
- SIBLING_OF, PARENT_OF, CHILD_OF, GRANDPARENT_OF, GRANDCHILD_OF
- RELATIVE_OF (for extended family: aunt, uncle, cousin)

Romantic:
- DATING, ENGAGED_TO, MARRIED_TO
- EX_PARTNER_OF (former romantic relationship)
- BROKE_UP_WITH (when breakup timing is mentioned)

Social & Professional:
- KNOWS, WORKS_AT, COWORKER_OF, FRIENDS_WITH, FAN_OF

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
1. Extract ONLY what is explicitly stated in the USER MESSAGE - do not infer
2. For temporal changes (was/now/planning), create SEPARATE relationships for each state
3. Include temporal context in metadata when mentioned (timeframe, reason, purpose)
4. Do NOT use generic types as names (e.g., use "Mom" instead of "Person", "Philadelphia" instead of "Place")
5. Return empty arrays if nothing found
6. CRITICAL: Only extract facts from the USER MESSAGE. Do NOT extract:
   - Suggestions, recommendations, or information provided by the ASSISTANT
   - Hypotheticals or examples given by the ASSISTANT
   - Places, restaurants, or entities that only appear in ASSISTANT responses
7. Attribution: The USER is always the subject of their own feelings.
   - "I feel upset about X" -> USER -FEELS_ABOUT-> X
   - Do NOT reverse the subject
8. Temporal metadata: Capture duration and timing when mentioned.
   - "6-year relationship" -> metadata: {{"duration": "6 years"}}
   - "broke up last week" -> metadata: {{"when": "last week"}}

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

Example 3 - State transition:
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

Example 4 - Family with explicit relationships:
Input: "My brother Sam loves 3D printing. My sister Justine is visiting."
Output:
{{
  "entities": [
    {{"name": "USER", "type": "Person", "attributes": {{}}}},
    {{"name": "Sam", "type": "Person", "attributes": {{}}}},
    {{"name": "Justine", "type": "Person", "attributes": {{}}}},
    {{"name": "3D printing", "type": "Technology", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "Sam", "predicate": "SIBLING_OF", "object": "USER", "metadata": {{"type": "brother"}}}},
    {{"subject": "Justine", "predicate": "SIBLING_OF", "object": "USER", "metadata": {{"type": "sister"}}}},
    {{"subject": "Sam", "predicate": "FAN_OF", "object": "3D printing", "metadata": {{}}}}
  ]
}}

Example 5 - Breakup scenario:
Input: "I'm sad about my ex-girlfriend Giana. We broke up last week after 6 years."
Output:
{{
  "entities": [
    {{"name": "USER", "type": "Person", "attributes": {{}}}},
    {{"name": "Giana", "type": "Person", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "USER", "predicate": "EX_PARTNER_OF", "object": "Giana", "metadata": {{"duration": "6 years"}}}},
    {{"subject": "USER", "predicate": "BROKE_UP_WITH", "object": "Giana", "metadata": {{"when": "last week"}}}},
    {{"subject": "USER", "predicate": "FEELS_ABOUT", "object": "breakup with Giana", "metadata": {{"emotion": "sad"}}}}
  ]
}}

COMMON ERRORS TO AVOID:

ERROR 1 - Wrong predicate for ex-partners:
BAD:  Giana -FRIENDS_WITH-> USER
GOOD: USER -EX_PARTNER_OF-> Giana

ERROR 2 - Reversed attribution:
BAD:  Giana -FEELS_ABOUT-> breaking up
GOOD: USER -FEELS_ABOUT-> breakup with Giana

ERROR 3 - Extracting assistant suggestions:
BAD:  Bella Vita -LOCATED_AT-> South Philly (from assistant recommendation)
GOOD: (do not extract - not a user fact)

ERROR 4 - Inventing predicates:
BAD:  USER -SOMEONE_REPRESENTING-> Justine
GOOD: Justine -SIBLING_OF-> USER
"""


ASSISTANT_EXTRACTION_PROMPT = """Extract what the ASSISTANT did in this response.

ASSISTANT RESPONSE:
{assistant_text}

USER MESSAGE (context for what ASSISTANT is responding to):
{user_text}

Output valid JSON only. The subject is ALWAYS "ASSISTANT":
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "ASSISTANT", "predicate": "action type", "object": "entity2", "metadata": {{}}}}
    ]
}}

ASSISTANT ACTION PREDICATES:

- RECOMMENDED - suggested a restaurant, product, service, or specific action
- SUGGESTED - proposed an approach, idea, or alternative
- EXPLAINED - provided information, explanation, or clarification about a topic
- ASKED_ABOUT - asked the user a clarifying question about something
- OFFERED - offered to help with something specific

EXTRACTION RULES:
1. Subject is ALWAYS "ASSISTANT" - never the user
2. Only extract substantive actions - not pleasantries or acknowledgments
3. Object should be the specific thing recommended/explained/asked about
4. Include relevant context in metadata (reason, category, etc.)
5. Return empty arrays if the assistant response is just acknowledgment or small talk

EXAMPLES:

Example 1 - Restaurant recommendation:
User: "I'm looking for a good Italian restaurant for a date"
Assistant: "I'd recommend Bella Vita in South Philly - they have amazing pasta and a romantic atmosphere!"
Output:
{{
  "entities": [
    {{"name": "Bella Vita", "type": "Place", "attributes": {{"category": "restaurant", "cuisine": "Italian"}}}}
  ],
  "relationships": [
    {{"subject": "ASSISTANT", "predicate": "RECOMMENDED", "object": "Bella Vita", "metadata": {{"reason": "romantic dinner", "location": "South Philly"}}}}
  ]
}}

Example 2 - Technical explanation:
User: "How does async/await work in Python?"
Assistant: "Async/await is Python's syntax for writing asynchronous code. The async keyword defines a coroutine..."
Output:
{{
  "entities": [
    {{"name": "async/await in Python", "type": "Concept", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "ASSISTANT", "predicate": "EXPLAINED", "object": "async/await in Python", "metadata": {{}}}}
  ]
}}

Example 3 - Clarifying question:
User: "I want to improve my morning routine"
Assistant: "What time do you usually wake up? And are you looking to be more productive or more relaxed?"
Output:
{{
  "entities": [],
  "relationships": [
    {{"subject": "ASSISTANT", "predicate": "ASKED_ABOUT", "object": "wake up time", "metadata": {{}}}},
    {{"subject": "ASSISTANT", "predicate": "ASKED_ABOUT", "object": "routine goals", "metadata": {{"options": "productive vs relaxed"}}}}
  ]
}}

Example 4 - Suggestion with approach:
User: "My code is running slowly"
Assistant: "I'd suggest profiling first to identify bottlenecks, then we can optimize the hot paths."
Output:
{{
  "entities": [
    {{"name": "code profiling", "type": "Concept", "attributes": {{}}}}
  ],
  "relationships": [
    {{"subject": "ASSISTANT", "predicate": "SUGGESTED", "object": "code profiling", "metadata": {{"purpose": "identify bottlenecks"}}}}
  ]
}}

Example 5 - Just acknowledgment (no extraction):
User: "Thanks for your help!"
Assistant: "You're welcome! Let me know if you need anything else."
Output:
{{
  "entities": [],
  "relationships": []
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
