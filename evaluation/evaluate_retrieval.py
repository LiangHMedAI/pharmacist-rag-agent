"""Offline retrieval evaluation. This script makes no LLM/API calls."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_service import source_name
from rag_core import search_knowledge_base


def evaluate(dataset_path):
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    hits = 0
    reciprocal_rank_total = 0.0

    for case in cases:
        results = search_knowledge_base(case["query"])
        expected = case["expected_source_contains"].lower()
        names = [source_name(item.document).lower() for item in results]
        rank = next((index for index, name in enumerate(names, 1) if expected in name), None)
        if rank:
            hits += 1
            reciprocal_rank_total += 1 / rank
        print(f"{'PASS' if rank else 'FAIL'} | {case['query']} | rank={rank or '-'} | {names}")

    count = len(cases)
    print(f"\nHit-rate@k: {hits / count:.3f}")
    print(f"MRR@k: {reciprocal_rank_total / count:.3f}")
    return 0 if hits == count else 1


if __name__ == "__main__":
    default_dataset = Path(__file__).with_name("golden_questions.example.json")
    raise SystemExit(evaluate(sys.argv[1] if len(sys.argv) > 1 else default_dataset))

