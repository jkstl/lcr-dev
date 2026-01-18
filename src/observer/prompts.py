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

Output valid JSON only. Use EXACTLY this schema (entities as array, relationships with "predicate" key):
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Technology|Place|Organization|Event|Concept", "attributes": {{}}}}
    ],
    "relationships": [
        {{"subject": "entity1", "predicate": "PREDICATE_NAME", "object": "entity2", "metadata": {{}}}}
    ]
}}

CRITICAL: Use "predicate" not "relation". Use arrays not dicts.

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

ASSISTANT ACTION PREDICATES (use ONLY these):

- RECOMMENDED - suggested a specific restaurant, product, service, or concrete action to take
- SUGGESTED - proposed a specific approach, solution, or alternative
- EXPLAINED - provided factual/technical information the user asked about
- ASKED_ABOUT - asked the user a specific clarifying question
- OFFERED - offered to perform a specific task for the user

WHAT IS NOT AN ACTION (return empty arrays for ALL of these):

- Empathy: "I'm sorry to hear that", "That must be hard", "Breakups are never easy"
- Acknowledgment: "That sounds nice", "I understand", "That sounds lovely"
- Reflecting: "It sounds like you're feeling...", "It sounds like you're second-guessing..."
- Open invitations: "Would you like to talk about it?", "Would you like to talk about what's been on your mind?"
- Analysis: "Given how quickly she responded, she seems comfortable", "she's at least comfortable with the exchange"
- Validation: "It's okay to feel sad", "That's understandable"

CRITICAL: "Would you like to talk about X?" is NOT ASKED_ABOUT - it's an open invitation.
CRITICAL: Analyzing someone's behavior is NOT EXPLAINED - no new information was provided.

EXTRACTION RULES:
1. Subject is ALWAYS "ASSISTANT" - never the user
2. ONLY extract if assistant gave CONCRETE, SPECIFIC, ACTIONABLE information
3. Empathy, validation, and emotional support are NOT actions - return empty arrays
4. "Would you like to talk about X?" is NOT an action - it's an open invitation
5. Object must be a specific named thing (restaurant name, technical concept, etc.)
6. Return empty arrays if unsure - false negatives are better than false positives

EXAMPLES OF ACTIONS (extract these):

Example 1 - Restaurant recommendation:
User: "I'm looking for a good Italian restaurant"
Assistant: "I'd recommend Bella Vita in South Philly - great pasta!"
Output:
{{
  "entities": [{{"name": "Bella Vita", "type": "Place", "attributes": {{"category": "restaurant"}}}}],
  "relationships": [{{"subject": "ASSISTANT", "predicate": "RECOMMENDED", "object": "Bella Vita", "metadata": {{}}}}]
}}

Example 2 - Technical explanation (user asked a question):
User: "How does async/await work in Python?"
Assistant: "Async/await is Python's syntax for asynchronous code..."
Output:
{{
  "entities": [{{"name": "async/await in Python", "type": "Concept", "attributes": {{}}}}],
  "relationships": [{{"subject": "ASSISTANT", "predicate": "EXPLAINED", "object": "async/await in Python", "metadata": {{}}}}]
}}

EXAMPLES OF NON-ACTIONS (return empty arrays):

Example 3 - Empathy for breakup (NO extraction):
User: "I broke up with my girlfriend and I'm feeling sad"
Assistant: "I'm really sorry you're going through this. Breakups are never easy. Would you like to talk about what's been on your mind?"
Output:
{{
  "entities": [],
  "relationships": []
}}

Example 4 - Acknowledging family visit (NO extraction):
User: "My mom and sister are coming to visit today"
Assistant: "That sounds lovely! It must mean a lot to have them traveling to see you."
Output:
{{
  "entities": [],
  "relationships": []
}}

Example 5 - Analysis without recommendation (NO extraction):
User: "I texted my ex and she replied quickly"
Assistant: "It sounds like she's comfortable with the exchange. Given how fast she responded, she probably doesn't mind hearing from you."
Output:
{{
  "entities": [],
  "relationships": []
}}

Example 6 - Open invitation to talk (NO extraction - NOT ASKED_ABOUT):
User: "I broke up with my girlfriend and I'm sad"
Assistant: "I'm sorry. Breakups are hard. Would you like to talk about what's been on your mind?"
Output:
{{
  "entities": [],
  "relationships": []
}}

Example 7 - Just acknowledgment (NO extraction):
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
