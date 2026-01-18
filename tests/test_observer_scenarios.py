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
            "expected_user_entities": ["Giana", "USER"],
            "expected_user_predicates": ["EX_PARTNER_OF", "BROKE_UP_WITH", "FEELS_ABOUT"],
            "forbidden_user_entities": [],
            "expected_assistant_predicates": [],  # Just empathy, no substantive action
        },
        {
            "name": "family_visit",
            "description": "User describes family visiting from Massachusetts",
            "user": "It's ok. My mom and sister are coming to see me in Philadelphia today, we are going out to dinner. They are traveling from West Boylston Massachusetts. My sisters name is Justine, shes 24",
            "assistant": "That sounds like a lovely way to spend the day — your mom and Justine traveling all that way to see you must mean a lot. It's sweet that you're sharing this time with them, especially after everything. I hope dinner is a nice distraction.",
            "expected_user_entities": ["Mom", "Justine", "Philadelphia", "West Boylston"],
            "expected_user_predicates": ["SIBLING_OF", "PARENT_OF"],
            "forbidden_user_entities": [],
            "expected_assistant_predicates": [],  # Just acknowledgment/empathy
        },
        {
            "name": "texted_giana",
            "description": "User mentions texting Giana",
            "user": "I sent Giana a text today saying 'have a nice weekend', not sure if that was the right thing to do, but she replied back 'you too' almost immediately",
            "assistant": "It sounds like you're second-guessing whether the message was too casual or if it might have stirred up anything from the past. Given how quickly she responded, it seems like she's at least comfortable with the exchange.",
            "expected_user_entities": ["Giana", "USER"],
            "expected_user_predicates": [],  # Action, not a relationship change
            "forbidden_user_entities": ["have a nice weekend", "you too"],  # These are messages, not entities
            "expected_assistant_predicates": [],  # Just analysis, no recommendation
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
            user_predicates = [r.get("predicate", "") for r in user_result.get("relationships", [])]

            # Check expected entities
            for expected in scenario.get("expected_user_entities", []):
                if expected.lower() not in user_entities:
                    errors.append(f"USER missing entity: {expected}")

            # Check expected predicates
            for expected in scenario.get("expected_user_predicates", []):
                if expected not in user_predicates:
                    errors.append(f"USER missing predicate: {expected}")

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

            # If no actions expected, should be empty
            if not scenario.get("expected_assistant_predicates"):
                if assistant_relationships:
                    errors.append(f"ASSISTANT should have no actions, got: {assistant_predicates}")

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
