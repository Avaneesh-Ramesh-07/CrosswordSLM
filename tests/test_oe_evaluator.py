"""Tests for the OpenEvolve evaluator core + entry point + harvest hook.
Run on Linux (WSL/Colab): python3 tests/test_oe_evaluator.py

The strict checks use a HARDCODER that returns a known-valid grid (the NYT
fixture), so they test the evaluator WIRING deterministically, independent of
the reference generator's fill reliability on constrained vocab. A separate
informational run exercises the real generator at size 7.
"""

import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, build_layout_from_grid  # noqa: E402
from pipeline import oe_evaluator as oe  # noqa: E402
from pipeline.word_source import index_by_length, load_scored_dict  # noqa: E402
from tests import fixtures  # noqa: E402
import seeds.reference_v1 as ref  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def nyt_layout_words_scores():
    layout = build_layout_from_grid(fixtures.grid_map(), fixtures.size())
    words = sorted({e["answer"] for e in layout["across"]} | {e["answer"] for e in layout["down"]})
    scores = {w: 70 for w in words}
    return layout, words, scores


def hardcoder_code(layout):
    return "import json\nLAYOUT = json.loads(r'''" + json.dumps(layout) + "''')\n" \
           "def generate_crossword(topic, word_source, size):\n    return LAYOUT\n"


BROKEN = """
def generate_crossword(topic, word_source, size):
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
"""


def main():
    random.seed(0)
    layout, words, scores = nyt_layout_words_scores()
    spec15 = Spec(size=15, require_symmetry=True)

    # --- core: a valid grid scores well; metrics expose feature dims ---
    out = oe.evaluate_code(hardcoder_code(layout), spec15, words, scores=scores, n_draws=2)
    m = out["metrics"]
    print("valid-grid metrics:", m)
    check("valid combined_score > 0.7", m["combined_score"] > 0.7, str(m))
    check("valid rate == 1.0", m["valid"] == 1.0, str(m))
    check("fill_quality reflects scores (~0.7)", abs(m["fill_quality"] - 0.7) < 0.05, str(m))
    check("feature dims present", "fill_density" in m and "coverage" in m)

    # --- core: broken generator scores zero ---
    outb = oe.evaluate_code(BROKEN, spec15, words, scores=scores, n_draws=1)
    check("broken combined_score == 0", outb["metrics"]["combined_score"] == 0.0, str(outb["metrics"]))
    check("broken valid rate == 0", outb["metrics"]["valid"] == 0.0)
    check("broken artifacts record failure", outb["artifacts"]["n_valid"] == 0)

    # --- entry point: problem JSON + env + harvest hook ---
    tmp = tempfile.mkdtemp()
    problem_path = os.path.join(tmp, "problem.json")
    harvest_path = os.path.join(tmp, "harvest.jsonl")
    hc_path = os.path.join(tmp, "cand.py")
    with open(hc_path, "w", encoding="utf-8") as fh:
        fh.write(hardcoder_code(layout))
    problem = {
        "spec_id": "s00042", "size": 15, "require_symmetry": True, "min_word_len": 3,
        "time_budget_s": 5, "density_target": 0.72, "topic_words": [],
        "word_source": words, "scores": scores, "n_draws": 1, "seed": 3,
    }
    with open(problem_path, "w", encoding="utf-8") as fh:
        json.dump(problem, fh)
    os.environ[oe.PROBLEM_ENV] = problem_path
    os.environ[oe.HARVEST_ENV] = harvest_path

    result = oe.evaluate(hc_path)
    combined = result["combined_score"] if isinstance(result, dict) else result.metrics["combined_score"]
    check("entry point returns combined_score", combined > 0.7, str(combined))
    rows = [json.loads(x) for x in open(harvest_path, encoding="utf-8")]
    check("harvest wrote one row", len(rows) == 1, str(len(rows)))
    check("harvest row has spec_id/code/metrics", rows[0]["spec_id"] == "s00042" and "code" in rows[0] and "metrics" in rows[0])

    # --- informational: real reference generator at size 7 (blocks fill robustly) ---
    fill = load_scored_dict(min_len=3, max_len=7)
    ws7 = [w for L in (3, 4, 5, 6, 7) for w, _ in index_by_length(fill).get(L, [])]
    ref_code = open(ref.__file__, encoding="utf-8").read()
    out7 = oe.evaluate_code(ref_code, Spec(size=7, require_symmetry=True, time_budget_s=3), ws7, scores=fill, n_draws=1)
    print("reference 7x7 metrics:", out7["metrics"])
    check("evaluator returns well-formed metrics for real generator", "combined_score" in out7["metrics"])

    print(f"\nAll {PASS} oe-evaluator checks passed.")


if __name__ == "__main__":
    main()
