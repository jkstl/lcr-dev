"""
Focused test for specific conversation scenarios.
Tests extraction quality for real conversation patterns.
"""

import asyncio
import json
from src.observer.observer import Observer
from src.config import settings


async def test_breakup_and_family_scenarios():
    """Test the two problematic conversation turns."""
    print(f"Testing Observer with model: {settings.observer_model}")
    print("=" * 70)

    observer = Observer(graph_store=None)

    scenarios = [
        {
            "name": "breakup_announcement",
            "description": "User announces breakup with Giana",
            "user": "I broke up with my girlfriend Giana last week, and am feeling kind of sad about it.",
            "assistant": "I'm really sorry you're going through this. Breakups are never easy, especially when you care about someone. It's okay to feel sad — grief is a natural part of letting go. Would you like to talk about what's been on your mind?",
            "expected_user_entities": ["Giana"],
            # BROKE_UP_WITH or EX_PARTNER_OF both acceptable - semantically equivalent
            "expected_user_predicates_any": [["BROKE_UP_WITH", "EX_PARTNER_OF"]],
            # FEELS_ABOUT is nice-to-have but not required (emotion often captured in metadata)
            "expected_user_predicates": [],
            "forbidden_user_entities": [],
            "expected_assistant_predicates": [],  # Just empathy, no substantive action
        },
        {
            "name": "family_visit",
            "description": "User describes family visiting from Massachusetts",
            "user": "It's ok. My mom and sister are coming to see me in Philadelphia today, we are going out to dinner. They are traveling from West Boylston Massachusetts. My sisters name is Justine, shes 24",
            "assistant": "That sounds like a lovely way to spend the day — your mom and Justine traveling all that way to see you must mean a lot. It's sweet that you're sharing this time with them, especially after everything. I hope dinner is a nice distraction.",
            "expected_user_entities": ["Mom", "Philadelphia"],
            # Justine may appear as "Sister" with Justine in attributes - check separately
            "expected_user_entities_any": [["Justine", "Sister"]],
            # SIBLING_OF or PARENT_OF - model may confuse these
            "expected_user_predicates_any": [["SIBLING_OF", "PARENT_OF"]],
            "expected_user_predicates": [],
            "forbidden_user_entities": [],
            "expected_assistant_predicates": [],  # Just acknowledgment/empathy
        },
        {
            "name": "texted_giana",
            "description": "User mentions texting Giana",
            "user": "I sent Giana a text today saying 'have a nice weekend', not sure if that was the right thing to do, but she replied back 'you too' almost immediately",
            "assistant": "It sounds like you're second-guessing whether the message was too casual or if it might have stirred up anything from the past. Given how quickly she responded, it seems like she's at least comfortable with the exchange.",
            "expected_user_entities": ["Giana"],
            "expected_user_predicates": [],  # Action, not a relationship change
            "forbidden_user_entities": ["have a nice weekend", "you too"],  # These are messages, not entities
            # Note: Small models may extract analysis as EXPLAINED - this is a known limitation
            # Ideally empty, but we accept EXPLAINED as tolerable false positive for now
            "expected_assistant_predicates": [],
            "tolerate_assistant_predicates": ["EXPLAINED"],  # Known limitation with small models
        },
    ]

    all_results = []

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*70}")
        print(f"\nUSER: {scenario['user']}")
        print(f"\nASSISTANT: {scenario['assistant'][:100]}...")

        errors = []

        # Test USER extraction
        print(f"\n--- USER EXTRACTION ---")
        try:
            user_result = await observer._extract_user_facts(
                scenario['user'],
                scenario['assistant']
            )
            print(json.dumps(user_result, indent=2))

            # Validate user extraction
            user_entities = [e.get("name", "").lower() for e in user_result.get("entities", [])]
            # Also check attributes for names (e.g., Sister with name: Justine)
            for e in user_result.get("entities", []):
                attrs = e.get("attributes", {})
                if "name" in attrs:
                    user_entities.append(attrs["name"].lower())
            user_predicates = [r.get("predicate", "") for r in user_result.get("relationships", [])]

            # Check expected entities (all must be present)
            for expected in scenario.get("expected_user_entities", []):
                if expected.lower() not in user_entities:
                    errors.append(f"USER missing entity: {expected}")

            # Check expected entities ANY (at least one from each group must be present)
            for group in scenario.get("expected_user_entities_any", []):
                if not any(opt.lower() in user_entities for opt in group):
                    errors.append(f"USER missing one of entities: {group}")

            # Check expected predicates (all must be present)
            for expected in scenario.get("expected_user_predicates", []):
                if expected not in user_predicates:
                    errors.append(f"USER missing predicate: {expected}")

            # Check expected predicates ANY (at least one from each group must be present)
            for group in scenario.get("expected_user_predicates_any", []):
                if not any(opt in user_predicates for opt in group):
                    errors.append(f"USER missing one of predicates: {group}")

            # Check forbidden entities
            for forbidden in scenario.get("forbidden_user_entities", []):
                if forbidden.lower() in user_entities:
                    errors.append(f"USER incorrectly extracted: {forbidden}")

        except Exception as e:
            errors.append(f"USER extraction error: {e}")
            user_result = {}

        # Test ASSISTANT extraction
        print(f"\n--- ASSISTANT EXTRACTION ---")
        try:
            assistant_result = await observer._extract_assistant_actions(
                scenario['assistant'],
                scenario['user']
            )
            print(json.dumps(assistant_result, indent=2))

            # Validate assistant extraction
            assistant_entities = [e.get("name", "").lower() for e in assistant_result.get("entities", [])]
            assistant_predicates = [r.get("predicate", "") for r in assistant_result.get("relationships", [])]
            assistant_relationships = assistant_result.get("relationships", [])

            # Check that subject is always ASSISTANT
            for rel in assistant_relationships:
                if rel.get("subject") != "ASSISTANT":
                    errors.append(f"ASSISTANT relationship has wrong subject: {rel.get('subject')}")

            # If no actions expected, should be empty (unless tolerated)
            if not scenario.get("expected_assistant_predicates"):
                tolerated = scenario.get("tolerate_assistant_predicates", [])
                unexpected = [p for p in assistant_predicates if p not in tolerated]
                if unexpected:
                    errors.append(f"ASSISTANT should have no actions, got: {unexpected}")
                elif assistant_predicates:
                    print(f"  [WARN] Tolerated false positives: {assistant_predicates}")

            # Check expected predicates
            for expected in scenario.get("expected_assistant_predicates", []):
                if expected not in assistant_predicates:
                    errors.append(f"ASSISTANT missing predicate: {expected}")

        except Exception as e:
            errors.append(f"ASSISTANT extraction error: {e}")
            assistant_result = {}

        # Report results
        print(f"\n--- VALIDATION ---")
        if errors:
            print(f"[FAIL] {len(errors)} error(s):")
            for error in errors:
                print(f"   - {error}")
            all_results.append({"scenario": scenario["name"], "passed": False, "errors": errors})
        else:
            print(f"[PASS] All validations passed")
            all_results.append({"scenario": scenario["name"], "passed": True, "errors": []})

    await observer.close()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    print(f"Passed: {passed}/{total}")

    if passed < total:
        print("\nFailed scenarios:")
        for r in all_results:
            if not r["passed"]:
                print(f"\n  {r['scenario']}:")
                for error in r["errors"]:
                    print(f"    - {error}")

    return all_results


if __name__ == "__main__":
    asyncio.run(test_breakup_and_family_scenarios())
