"""
RAG Quality Tests — automated evaluation of the SUT RAG system.

Metrics computed:
  1. Keyword Coverage   : does the answer contain expected domain keywords?
  2. Context Relevance  : does the answer reference retrieved context?
  3. Answer Length      : proxy for completeness
  4. Source Citation    : does the answer cite SUT article numbers?
  5. Faithfulness Score : fraction of expected context found in answer (soft match)

These tests call the running backend at BASE_URL (defaults to localhost:8000).
Set RAG_TEST_USER / RAG_TEST_PASSWORD env vars or use the defaults below.

Usage:
  pytest rag_quality/test_rag_quality.py -v -m slow
"""
import os
import sys
import json
import math
import time
import statistics
import re
from pathlib import Path
from typing import List, Dict, Any

import pytest
import httpx

BASE_URL         = os.getenv("RAG_BASE_URL", "http://localhost:8000")
RAG_USERNAME     = os.getenv("RAG_TEST_USER", "admin")
RAG_PASSWORD     = os.getenv("RAG_TEST_PASSWORD", "Admin@1234!")
QUESTIONS_FILE   = Path(__file__).parent / "sample_questions.json"
RESULTS_FILE     = Path(__file__).parent / "rag_quality_results.json"

pytestmark = pytest.mark.slow


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def auth_token():
    """Login and get a JWT."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        r = client.post("/api/auth/login", data={
            "username": RAG_USERNAME,
            "password": RAG_PASSWORD,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        if r.status_code != 200:
            pytest.skip(f"Cannot login to RAG backend ({r.status_code}). "
                        "Start docker compose and ensure credentials are correct.")
        return r.json()["access_token"]


@pytest.fixture(scope="module")
def questions():
    if not QUESTIONS_FILE.exists():
        pytest.skip(f"Questions file not found: {QUESTIONS_FILE}")
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rag_results(auth_token, questions) -> List[Dict[str, Any]]:
    """
    Run all questions through the RAG system and collect answers.
    Results are cached to RESULTS_FILE for inspection.
    """
    results = []
    headers = {"Authorization": f"Bearer {auth_token}"}

    with httpx.Client(base_url=BASE_URL, timeout=120) as client:
        for q in questions:
            start = time.time()
            final_answer = ""
            try:
                with client.stream("POST", "/api/chat/query",
                                   json={"query": q["question"], "role": "PATIENT"},
                                   headers=headers) as resp:
                    if resp.status_code != 200:
                        results.append({**q, "answer": "", "latency": 0,
                                        "error": f"HTTP {resp.status_code}"})
                        continue
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                if "final_answer" in chunk:
                                    final_answer = chunk["final_answer"]
                                    break
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                results.append({**q, "answer": "", "latency": 0, "error": str(e)})
                continue

            latency = time.time() - start
            results.append({**q, "answer": final_answer, "latency": latency, "error": None})

    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def keyword_coverage(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def has_source_citation(answer: str) -> bool:
    """Check if answer cites a SUT article number like [Madde X.X.X]."""
    return bool(re.search(r'\[?[Mm]adde\s+\d+[\.\d]*\]?', answer))


def answer_length_score(answer: str, min_chars: int = 200) -> float:
    """Score based on answer length (completeness proxy)."""
    length = len(answer.strip())
    return min(1.0, length / min_chars)


def faithfulness_score(answer: str, expected_context: List[str]) -> float:
    """Soft match: how many expected context terms appear in the answer."""
    if not expected_context:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for term in expected_context if term.lower() in answer_lower)
    return hits / len(expected_context)


# ─────────────────────────────────────────────────────────────────────────────
# Per-question Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGAnswerQuality:
    """One test per quality metric across all questions."""

    def test_all_questions_get_answers(self, rag_results):
        """Every question must receive a non-empty answer."""
        failed = [r["id"] for r in rag_results if not r.get("answer")]
        assert not failed, f"Questions with empty answers: {failed}"

    def test_no_api_errors(self, rag_results):
        """No question should fail with an API error."""
        errors = [(r["id"], r.get("error")) for r in rag_results if r.get("error")]
        assert not errors, f"Questions with errors: {errors}"

    def test_average_keyword_coverage_above_threshold(self, rag_results):
        """Average keyword coverage across all questions must be >= 0.40."""
        scores = [
            keyword_coverage(r["answer"], r.get("expected_keywords", []))
            for r in rag_results if r.get("answer")
        ]
        avg = statistics.mean(scores) if scores else 0
        print(f"\n  Average keyword coverage: {avg:.2%}")
        assert avg >= 0.40, f"Keyword coverage too low: {avg:.2%} (min 40%)"

    def test_average_faithfulness_above_threshold(self, rag_results):
        """Average faithfulness score must be >= 0.40."""
        scores = [
            faithfulness_score(r["answer"], r.get("expected_context_contains", []))
            for r in rag_results if r.get("answer")
        ]
        avg = statistics.mean(scores) if scores else 0
        print(f"\n  Average faithfulness score: {avg:.2%}")
        assert avg >= 0.40, f"Faithfulness too low: {avg:.2%} (min 40%)"

    def test_majority_answers_have_source_citations(self, rag_results):
        """At least 50% of answers should cite a SUT article number."""
        answers_with_answers = [r for r in rag_results if r.get("answer")]
        cited = sum(1 for r in answers_with_answers if has_source_citation(r["answer"]))
        rate = cited / len(answers_with_answers) if answers_with_answers else 0
        print(f"\n  Source citation rate: {rate:.2%}")
        assert rate >= 0.50, f"Citation rate too low: {rate:.2%} (min 50%)"

    def test_answers_are_sufficiently_long(self, rag_results):
        """Average answer length should be >= 150 characters."""
        lengths = [len(r["answer"].strip()) for r in rag_results if r.get("answer")]
        avg = statistics.mean(lengths) if lengths else 0
        print(f"\n  Average answer length: {avg:.0f} chars")
        assert avg >= 150, f"Answers too short on average: {avg:.0f} chars (min 150)"

    def test_p95_latency_under_60s(self, rag_results):
        """95th percentile latency must be under 60 seconds."""
        latencies = sorted([r["latency"] for r in rag_results if r.get("answer")])
        if not latencies:
            pytest.skip("No latency data")
        p95_idx = math.ceil(0.95 * len(latencies)) - 1
        p95 = latencies[p95_idx]
        print(f"\n  P95 latency: {p95:.1f}s")
        assert p95 < 60, f"P95 latency too high: {p95:.1f}s (max 60s)"


# ─────────────────────────────────────────────────────────────────────────────
# Per-category Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGByCategory:
    """Validate that different question categories all get reasonable coverage."""

    def _category_results(self, rag_results, category):
        return [r for r in rag_results if r.get("category") == category and r.get("answer")]

    def test_drug_coverage_questions(self, rag_results):
        items = self._category_results(rag_results, "drug_coverage")
        if not items:
            pytest.skip("No drug_coverage results")
        scores = [keyword_coverage(r["answer"], r.get("expected_keywords", [])) for r in items]
        assert statistics.mean(scores) >= 0.30, "Drug coverage questions underperforming"

    def test_specialist_report_questions(self, rag_results):
        items = self._category_results(rag_results, "specialist_report")
        if not items:
            pytest.skip("No specialist_report results")
        scores = [keyword_coverage(r["answer"], r.get("expected_keywords", [])) for r in items]
        assert statistics.mean(scores) >= 0.30, "Specialist report questions underperforming"

    def test_device_coverage_questions(self, rag_results):
        items = self._category_results(rag_results, "device_coverage")
        if not items:
            pytest.skip("No device_coverage results")
        scores = [keyword_coverage(r["answer"], r.get("expected_keywords", [])) for r in items]
        assert statistics.mean(scores) >= 0.30, "Device coverage questions underperforming"


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report (printed to console)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGSummaryReport:
    """Print a summary table after all quality tests run."""

    def test_print_full_summary(self, rag_results):
        """Not a real assertion — prints a table for the report."""
        print("\n" + "=" * 70)
        print(f"{'ID':<8} {'KW%':>6} {'Faith%':>8} {'Cite':>6} {'Len':>6} {'Sec':>6}")
        print("-" * 70)
        for r in rag_results:
            ans = r.get("answer", "")
            kw = keyword_coverage(ans, r.get("expected_keywords", []))
            fa = faithfulness_score(ans, r.get("expected_context_contains", []))
            ci = "✓" if has_source_citation(ans) else "✗"
            le = len(ans.strip())
            lt = r.get("latency", 0)
            err = r.get("error", "")
            row = f"{r['id']:<8} {kw:>6.0%} {fa:>8.0%} {ci:>6} {le:>6} {lt:>5.1f}s"
            if err:
                row += f"  ⚠ {err}"
            print(row)
        print("=" * 70)
        assert True  # always pass — this is just reporting
