"""
Test cases for Observer extraction quality.
Tests the 6 critical failure scenarios identified in chatlog.md analysis.
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


async def test_extraction():
    """Run extraction tests for all failure scenarios."""
    print(f"Testing Observer with model: {settings.observer_model}")
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

        combined = f"USER: {scenario['user']}\nASSISTANT: {scenario['assistant']}"

        try:
            result = await observer._extract_entities(combined)
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
    print("SUMMARY")
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


async def test_single_scenario(scenario_name: str):
    """Test a single scenario by name."""
    observer = Observer(graph_store=None)

    scenarios = {
        "breakup": {
            "user": "yeah it will help take my mind off my ex-girlfriend, Giana. Her and I broke up last week after a six year relationship",
            "assistant": "I'm really sorry to hear about your breakup..."
        },
        "sibling": {
            "user": "my brother Sam is really into 3-d printing",
            "assistant": "That's awesome!"
        }
    }

    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        return

    s = scenarios[scenario_name]
    combined = f"USER: {s['user']}\nASSISTANT: {s['assistant']}"
    result = await observer._extract_entities(combined)
    print(json.dumps(result, indent=2))

    await observer.close()


if __name__ == "__main__":
    asyncio.run(test_extraction())
