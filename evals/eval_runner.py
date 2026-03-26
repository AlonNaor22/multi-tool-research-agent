"""Evaluation runner for the Multi-Tool Research Agent.

Tests the agent's decision-making: does it pick the right tool and
produce a correct answer?

Usage:
    python -m evals.eval_runner                          # run all cases
    python -m evals.eval_runner --case calc_percentage   # run one case
    python -m evals.eval_runner --category "MATH"        # run one category
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable debug/verbose BEFORE importing the agent (it reads VERBOSE at init)
import config
config.VERBOSE = False

from src.agent import ResearchAgent
from evals.eval_callback import EvalCallbackHandler

# Fix Windows console encoding for emoji output
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )


# ── Loading ──────────────────────────────────────────────────────────

CASES_FILE = Path(__file__).parent / "test_cases.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_test_cases(
    case_id: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Load test cases, optionally filtering by id or category."""
    with open(CASES_FILE) as f:
        cases = json.load(f)

    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            print(f"Error: No test case with id '{case_id}'")
            sys.exit(1)

    if category:
        cat_upper = category.upper()
        cases = [c for c in cases if cat_upper in c["category"].upper()]
        if not cases:
            print(f"Error: No test cases matching category '{category}'")
            sys.exit(1)

    return cases


# ── Scoring ──────────────────────────────────────────────────────────

def score_tool_selection(
    tools_called: list[str],
    expected_tools: list[str],
) -> bool:
    """Did the agent call at least one of the expected tools?"""
    return any(tool in tools_called for tool in expected_tools)


def score_answer(answer: str, expected_keywords: list[str]) -> bool:
    """Do ALL expected keywords appear in the answer (case-insensitive)?

    Handles number formatting: strips commas from both answer and keywords
    so "5,280" matches "5280" and vice versa.
    """
    answer_normalized = answer.lower().replace(",", "")
    return all(
        kw.lower().replace(",", "") in answer_normalized
        for kw in expected_keywords
    )


# ── Runner ───────────────────────────────────────────────────────────

def run_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all test cases through the agent and collect results."""
    print("\n" + "=" * 65)
    print("  EVALUATION SUITE — Multi-Tool Research Agent")
    print("=" * 65)
    print(f"  Running {len(cases)} test case(s)...\n")

    # Create a fresh agent (no memory carryover between cases)
    # config.VERBOSE is already set to False at module level
    agent = ResearchAgent()

    # Our eval callback to track tool selections
    eval_cb = EvalCallbackHandler()

    results = []
    total_start = time.time()

    for i, case in enumerate(cases, 1):
        case_id = case["id"]
        question = case["question"]
        expected_tools = case["expected_tools"]
        expected_keywords = case["expected_keywords"]
        category = case["category"]

        print(f"  [{i}/{len(cases)}] {case_id}")
        print(f"         Q: {question}")

        # Reset callback and memory between cases
        eval_cb.reset()
        agent.memory.clear()

        # Run the query — inject our eval callback
        case_start = time.time()
        answer = agent.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"callbacks": [eval_cb, agent.timing_callback],
             "recursion_limit": 20,
             "debug": False},
        )

        # Extract the text answer
        answer_text = agent._extract_answer(answer)
        duration = time.time() - case_start
        tools_called = eval_cb.get_tools_called()

        # Score
        tool_pass = score_tool_selection(tools_called, expected_tools)
        answer_pass = score_answer(answer_text, expected_keywords)

        status = "PASS" if (tool_pass and answer_pass) else "FAIL"
        tool_icon = "✅" if tool_pass else "❌"
        answer_icon = "✅" if answer_pass else "❌"

        print(f"         Tools called: {tools_called}")
        print(f"         Tool selection: {tool_icon}  (expected any of {expected_tools})")
        print(f"         Answer check:   {answer_icon}  (keywords: {expected_keywords})")
        print(f"         Time: {duration:.1f}s  |  Status: {status}")
        print()

        results.append({
            "id": case_id,
            "category": category,
            "question": question,
            "expected_tools": expected_tools,
            "expected_keywords": expected_keywords,
            "tools_called": tools_called,
            "tool_pass": tool_pass,
            "answer_pass": answer_pass,
            "answer_preview": answer_text[:200],
            "duration_s": round(duration, 2),
            "errors": eval_cb.errors,
        })

    total_duration = time.time() - total_start

    # ── Scorecard ────────────────────────────────────────────────────
    tool_passes = sum(1 for r in results if r["tool_pass"])
    answer_passes = sum(1 for r in results if r["answer_pass"])
    full_passes = sum(1 for r in results if r["tool_pass"] and r["answer_pass"])
    total = len(results)

    print("=" * 65)
    print("  SCORECARD")
    print("=" * 65)
    print(f"  Tool selection : {tool_passes}/{total} ({100 * tool_passes / total:.0f}%)")
    print(f"  Answer quality : {answer_passes}/{total} ({100 * answer_passes / total:.0f}%)")
    print(f"  Full pass      : {full_passes}/{total} ({100 * full_passes / total:.0f}%)")
    print(f"  Total time     : {total_duration:.1f}s")
    print("=" * 65)

    # Show failures
    failures = [r for r in results if not (r["tool_pass"] and r["answer_pass"])]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            reasons = []
            if not f["tool_pass"]:
                reasons.append(f"wrong tools (got {f['tools_called']}, expected any of {f['expected_tools']})")
            if not f["answer_pass"]:
                reasons.append(f"missing keywords {f['expected_keywords']}")
            print(f"    - {f['id']}: {', '.join(reasons)}")
    print()

    return {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total,
        "tool_accuracy": round(tool_passes / total, 3),
        "answer_accuracy": round(answer_passes / total, 3),
        "full_pass_rate": round(full_passes / total, 3),
        "total_duration_s": round(total_duration, 2),
        "results": results,
    }


def save_results(report: dict[str, Any]) -> Path:
    """Save the eval report as a timestamped JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    return filepath


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for the Multi-Tool Research Agent"
    )
    parser.add_argument(
        "--case",
        help="Run a single test case by ID (e.g. calc_percentage)",
    )
    parser.add_argument(
        "--category",
        help="Run all cases matching a category (e.g. MATH, WEATHER)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )
    args = parser.parse_args()

    cases = load_test_cases(case_id=args.case, category=args.category)
    report = run_eval(cases)

    if not args.no_save:
        filepath = save_results(report)
        print(f"  Results saved to: {filepath}\n")


if __name__ == "__main__":
    main()
