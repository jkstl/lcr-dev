"""
Test cases for Observer extraction quality.
Tests the 6 critical failure scenarios identified in chatlog.md analysis.
Also tests the dual extraction architecture for USER facts and ASSISTANT actions.
"""

import asyncio
import json
from src.observer.observer import Observer
from src.config import settings


# Expected extraction patterns for validation
EXPECTED_PATTERNS = {
    "breakup": {
        "required_predicates": ["EX_PARTNER_OF", "BROKE_UP_WITH"],
        "forbidden_predicates": ["FRIENDS_WITH"],
        "required_entities": ["Giana", "USER"],
    },
    "sibling": {
        "required_predicates": ["SIBLING_OF"],
        "forbidden_predicates": ["SOMEONE_REPRESENTING"],
        "required_entities": ["Sam", "USER"],
    },
    "feelings": {
        "required_subject": "USER",  # USER must be subject of FEELS_ABOUT
        "predicate": "FEELS_ABOUT",
    },
    "assistant_suggestion": {
        "forbidden_entities": ["Bella Vita", "South Philly restaurant"],
    },
}

# Expected patterns for assistant action extraction
ASSISTANT_PATTERNS = {
    "recommendation": {
        "required_predicates": ["RECOMMENDED"],
        "required_subject": "ASSISTANT",
    },
    "explanation": {
        "required_predicates": ["EXPLAINED"],
        "required_subject": "ASSISTANT",
    },
    "question": {
        "required_predicates": ["ASKED_ABOUT"],
        "required_subject": "ASSISTANT",
    },
}


def validate_extraction(result: dict, scenario_name: str) -> list[str]:
    """Validate extraction result against expected patterns."""
    errors = []

    if not result:
        return ["No extraction result"]

    entities = result.get("entities", [])
    relationships = result.get("relationships", [])

    entity_names = [e.get("name", "").lower() for e in entities]
    predicates = [r.get("predicate", "") for r in relationships]

    # Check based on scenario
    if scenario_name == "breakup":
        patterns = EXPECTED_PATTERNS["breakup"]
        for pred in patterns["required_predicates"]:
            if pred not in predicates:
                errors.append(f"Missing required predicate: {pred}")
        for pred in patterns["forbidden_predicates"]:
            if pred in predicates:
                errors.append(f"Should not use predicate: {pred}")
        # Check that Giana is extracted
        if "giana" not in entity_names:
            errors.append("Missing entity: Giana")

    elif scenario_name == "sibling":
        patterns = EXPECTED_PATTERNS["sibling"]
        if "SIBLING_OF" not in predicates:
            errors.append("Missing SIBLING_OF predicate for sibling relationship")
        for pred in patterns["forbidden_predicates"]:
            if pred in predicates:
                errors.append(f"Invented predicate used: {pred}")

    elif scenario_name == "feelings":
        # Check USER is subject of any FEELS_ABOUT relationship
        for rel in relationships:
            if rel.get("predicate") == "FEELS_ABOUT":
                if rel.get("subject") != "USER":
                    errors.append(f"FEELS_ABOUT has wrong subject: {rel.get('subject')} (should be USER)")

    elif scenario_name == "assistant_suggestion":
        patterns = EXPECTED_PATTERNS["assistant_suggestion"]
        for forbidden in patterns["forbidden_entities"]:
            if forbidden.lower() in entity_names:
                errors.append(f"Extracted assistant suggestion as entity: {forbidden}")

    return errors


async def test_user_extraction():
    """Run extraction tests for user fact scenarios."""
    print(f"Testing Observer USER extraction with model: {settings.observer_model}")
    print("=" * 60)

    observer = Observer(graph_store=None)

    scenarios = [
        {
            "name": "breakup",
            "description": "Ex-girlfriend extraction (Giana bug)",
            "user": "yeah it will help take my mind off my ex-girlfriend, Giana. Her and I broke up last week after a six year relationship",
            "assistant": "I'm really sorry to hear about your breakup with Giana...",
            "expected": "USER -EX_PARTNER_OF-> Giana, USER -BROKE_UP_WITH-> Giana (NOT FRIENDS_WITH)"
        },
        {
            "name": "sibling",
            "description": "Sibling relationship (Sam bug)",
            "user": "my brother Sam is really into 3-d printing, he can make phone cases and christmas decorations, its cool",
            "assistant": "That's awesome! Sam sounds like a total wizard with 3D printing...",
            "expected": "Sam -SIBLING_OF-> USER with metadata type: brother"
        },
        {
            "name": "sibling_justine",
            "description": "Sibling relationship (Justine bug)",
            "user": "my sister Justine is coming to visit next month",
            "assistant": "That's great that Justine is visiting!",
            "expected": "Justine -SIBLING_OF-> USER with metadata type: sister"
        },
        {
            "name": "feelings",
            "description": "Feelings attribution (perspective bug)",
            "user": "I'm feeling really upset about the whole situation with Giana",
            "assistant": "It's completely understandable to feel upset...",
            "expected": "USER -FEELS_ABOUT-> situation with Giana (USER as subject, not Giana)"
        },
        {
            "name": "assistant_suggestion",
            "description": "Assistant suggestion filtering (South Philly bug)",
            "user": "I'm looking for a good Italian restaurant",
            "assistant": "I'd recommend Bella Vita in South Philly - they have amazing pasta!",
            "expected": "Should NOT extract Bella Vita or South Philly (assistant suggestions)"
        },
        {
            "name": "temporal_metadata",
            "description": "Temporal metadata capture (duration bug)",
            "user": "We were together for 6 years before the breakup last week",
            "assistant": "Six years is a significant relationship...",
            "expected": "metadata should include duration: '6 years' and when: 'last week'"
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['description']} ---")
        print(f"Input: \"{scenario['user'][:60]}...\"")
        print(f"Expected: {scenario['expected']}")

        try:
            # Use the new _extract_user_facts method
            result = await observer._extract_user_facts(scenario['user'], scenario['assistant'])
            print(f"\nExtracted:")
            print(json.dumps(result, indent=2))

            # Validate the extraction
            errors = validate_extraction(result, scenario["name"])

            if errors:
                print(f"\n[FAIL] VALIDATION ERRORS:")
                for error in errors:
                    print(f"   - {error}")
                results.append({"scenario": scenario["name"], "passed": False, "errors": errors})
            else:
                print(f"\n[PASS] Extraction looks correct")
                results.append({"scenario": scenario["name"], "passed": True, "errors": []})

        except Exception as e:
            print(f"Error: {e}")
            results.append({"scenario": scenario["name"], "passed": False, "errors": [str(e)]})

    await observer.close()

    # Summary
    print("\n" + "=" * 60)
    print("USER EXTRACTION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed < total:
        print("\nFailed scenarios:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['scenario']}: {', '.join(r['errors'])}")

    return results


async def test_assistant_extraction():
    """Run extraction tests for assistant action scenarios."""
    print(f"\nTesting Observer ASSISTANT extraction with model: {settings.observer_model}")
    print("=" * 60)

    observer = Observer(graph_store=None)

    scenarios = [
        {
            "name": "recommendation",
            "description": "Restaurant recommendation",
            "user": "I'm looking for a good Italian restaurant for a date",
            "assistant": "I'd recommend Bella Vita in South Philly - they have amazing pasta and a romantic atmosphere!",
            "expected": "ASSISTANT -RECOMMENDED-> Bella Vita"
        },
        {
            "name": "explanation",
            "description": "Technical explanation",
            "user": "How does async/await work in Python?",
            "assistant": "Async/await is Python's syntax for writing asynchronous code. The async keyword defines a coroutine function, and await is used to pause execution until a result is ready.",
            "expected": "ASSISTANT -EXPLAINED-> async/await in Python"
        },
        {
            "name": "question",
            "description": "Clarifying question",
            "user": "I want to improve my morning routine",
            "assistant": "What time do you usually wake up? And are you looking to be more productive or more relaxed in the mornings?",
            "expected": "ASSISTANT -ASKED_ABOUT-> wake up time, routine goals"
        },
        {
            "name": "suggestion",
            "description": "Approach suggestion",
            "user": "My code is running slowly",
            "assistant": "I'd suggest profiling your code first to identify the bottlenecks, then we can focus on optimizing the hot paths.",
            "expected": "ASSISTANT -SUGGESTED-> code profiling"
        },
        {
            "name": "acknowledgment_only",
            "description": "Just acknowledgment (should extract nothing)",
            "user": "Thanks for your help!",
            "assistant": "You're welcome! Let me know if you need anything else.",
            "expected": "Empty extraction (no substantive action)"
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['description']} ---")
        print(f"Assistant: \"{scenario['assistant'][:60]}...\"")
        print(f"Expected: {scenario['expected']}")

        try:
            # Use the new _extract_assistant_actions method
            result = await observer._extract_assistant_actions(scenario['assistant'], scenario['user'])
            print(f"\nExtracted:")
            print(json.dumps(result, indent=2))

            # Validate the extraction
            errors = validate_assistant_extraction(result, scenario["name"])

            if errors:
                print(f"\n[FAIL] VALIDATION ERRORS:")
                for error in errors:
                    print(f"   - {error}")
                results.append({"scenario": scenario["name"], "passed": False, "errors": errors})
            else:
                print(f"\n[PASS] Extraction looks correct")
                results.append({"scenario": scenario["name"], "passed": True, "errors": []})

        except Exception as e:
            print(f"Error: {e}")
            results.append({"scenario": scenario["name"], "passed": False, "errors": [str(e)]})

    await observer.close()

    # Summary
    print("\n" + "=" * 60)
    print("ASSISTANT EXTRACTION SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed < total:
        print("\nFailed scenarios:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['scenario']}: {', '.join(r['errors'])}")

    return results


def validate_assistant_extraction(result: dict, scenario_name: str) -> list[str]:
    """Validate assistant action extraction result."""
    errors = []

    if not result:
        return ["No extraction result"]

    relationships = result.get("relationships", [])
    predicates = [r.get("predicate", "") for r in relationships]
    subjects = [r.get("subject", "") for r in relationships]

    if scenario_name == "acknowledgment_only":
        # Should have no relationships for pure acknowledgment
        if relationships:
            errors.append(f"Should not extract anything for acknowledgment, got: {predicates}")
        return errors

    # Check that subject is always ASSISTANT
    for rel in relationships:
        if rel.get("subject") != "ASSISTANT":
            errors.append(f"Subject should be ASSISTANT, got: {rel.get('subject')}")

    # Check based on scenario
    if scenario_name in ASSISTANT_PATTERNS:
        patterns = ASSISTANT_PATTERNS[scenario_name]
        for pred in patterns.get("required_predicates", []):
            if pred not in predicates:
                errors.append(f"Missing required predicate: {pred}")

    return errors


async def test_extraction():
    """Run all extraction tests (both user and assistant)."""
    user_results = await test_user_extraction()
    assistant_results = await test_assistant_extraction()

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    all_results = user_results + assistant_results
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    print(f"Total Passed: {passed}/{total}")

    return all_results


async def test_single_scenario(scenario_name: str):
    """Test a single scenario by name."""
    observer = Observer(graph_store=None)

    user_scenarios = {
        "breakup": {
            "user": "yeah it will help take my mind off my ex-girlfriend, Giana. Her and I broke up last week after a six year relationship",
            "assistant": "I'm really sorry to hear about your breakup..."
        },
        "sibling": {
            "user": "my brother Sam is really into 3-d printing",
            "assistant": "That's awesome!"
        }
    }

    assistant_scenarios = {
        "recommendation": {
            "user": "I'm looking for a good Italian restaurant for a date",
            "assistant": "I'd recommend Bella Vita in South Philly - they have amazing pasta!"
        },
        "explanation": {
            "user": "How does async/await work?",
            "assistant": "Async/await is syntax for writing asynchronous code..."
        }
    }

    if scenario_name in user_scenarios:
        s = user_scenarios[scenario_name]
        print(f"Testing USER extraction for: {scenario_name}")
        result = await observer._extract_user_facts(s['user'], s['assistant'])
        print(json.dumps(result, indent=2))
    elif scenario_name in assistant_scenarios:
        s = assistant_scenarios[scenario_name]
        print(f"Testing ASSISTANT extraction for: {scenario_name}")
        result = await observer._extract_assistant_actions(s['assistant'], s['user'])
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available user scenarios: {list(user_scenarios.keys())}")
        print(f"Available assistant scenarios: {list(assistant_scenarios.keys())}")

    await observer.close()


async def test_dual_extraction():
    """Test the full dual extraction flow (both user and assistant in one turn)."""
    print(f"\nTesting DUAL extraction with model: {settings.observer_model}")
    print("=" * 60)

    observer = Observer(graph_store=None)

    # Test case: User asks for restaurant, assistant recommends
    user_msg = "I'm looking for a good Italian restaurant. My girlfriend and I are celebrating our anniversary."
    assistant_msg = "I'd recommend Bella Vita in South Philly - they have amazing pasta and a romantic atmosphere perfect for anniversaries!"

    print(f"User: {user_msg}")
    print(f"Assistant: {assistant_msg}")
    print()

    # Extract user facts
    print("--- USER FACTS ---")
    user_result = await observer._extract_user_facts(user_msg, assistant_msg)
    print(json.dumps(user_result, indent=2))

    # Extract assistant actions
    print("\n--- ASSISTANT ACTIONS ---")
    assistant_result = await observer._extract_assistant_actions(assistant_msg, user_msg)
    print(json.dumps(assistant_result, indent=2))

    # Validate
    print("\n--- VALIDATION ---")
    user_entities = [e.get("name", "").lower() for e in user_result.get("entities", [])]
    assistant_entities = [e.get("name", "").lower() for e in assistant_result.get("entities", [])]

    # User facts should include girlfriend but NOT Bella Vita
    if "bella vita" in user_entities:
        print("[FAIL] User extraction incorrectly included Bella Vita (assistant suggestion)")
    else:
        print("[PASS] User extraction correctly excluded Bella Vita")

    # Assistant actions should include Bella Vita recommendation
    assistant_predicates = [r.get("predicate") for r in assistant_result.get("relationships", [])]
    if "RECOMMENDED" in assistant_predicates:
        print("[PASS] Assistant extraction correctly identified RECOMMENDED action")
    else:
        print("[FAIL] Assistant extraction missing RECOMMENDED predicate")

    await observer.close()


if __name__ == "__main__":
    asyncio.run(test_extraction())
