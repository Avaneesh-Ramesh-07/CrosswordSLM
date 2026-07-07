"""Verify the reference generator produces valid crosswords through the harness.

Run on Linux (WSL/Colab): python3 tests/test_reference_gen.py
Proves the POSITIVE case the earlier tests didn't: the harness accepts a real,
non-hardcoded generator that fills from the supplied word_source.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score  # noqa: E402
from harness.verify import fuzz_verify  # noqa: E402
from pipeline.word_source import index_by_length, load_scored_dict  # noqa: E402
import seeds.reference_v1 as ref  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def render(layout):
    size = layout["rows"]
    g = [["#"] * size for _ in range(size)]
    for cell in layout["cells"]:
        g[cell["r"]][cell["c"]] = cell["letter"]
    return "\n".join(" ".join(row) for row in g)


def main():
    random.seed(0)
    fill = load_scored_dict(min_len=3, max_len=7)  # default min_score=55
    idx = index_by_length(fill)
    five = [w for w, _ in idx[5]]
    print(f"fill dict: {len(fill):,} words (min_score=55); {len(five):,} five-letter")

    # --- direct run at size 5: render + score (with real fill-quality scores) ---
    words5 = random.sample(five, min(6000, len(five)))
    scores5 = {w: fill[w] for w in words5}
    layout = ref.generate_crossword("general", words5, 5)
    r = score(layout, Spec(size=5, require_symmetry=True), words5, scores=scores5)
    print("\nSample 5x5 fill:\n" + render(layout)
          + f"\n-> valid={r['valid']} score={r['combined_score']} density={r['fill_density']} fill_quality={r['fill_quality']}\n")
    check("direct 5x5 valid", r["valid"] == 1, str(r["reasons"]))
    check("direct 5x5 score high", r["combined_score"] >= 0.85, str(r["combined_score"]))

    # --- fuzz-verify across draws with DIFFERENT word_source subsets (robustness) ---
    code = open(ref.__file__, encoding="utf-8").read()
    draws = []
    for _ in range(3):
        subset = random.sample(five, min(5000, len(five)))
        draws.append((Spec(size=5, require_symmetry=True), subset))
    result = fuzz_verify(code, draws, timeout_s=10, scores=fill)
    print(f"fuzz-verify 5x5 over {result['n']} varied draws: accepted={result['accepted']} "
          f"n_valid={result['n_valid']} mean={result['mean_score']}")
    for i, rr in enumerate(result["results"]):
        print(f"  draw {i}: status={rr['status']} valid={rr.get('valid')} score={rr.get('combined_score')} runtime={rr.get('runtime_s')}s")
    check("fuzz 5x5 accepted (valid on all varied draws)", result["accepted"] is True, str(result))

    # --- best-effort size 7 (blocks); report only ---
    words7 = [w for w, _ in idx.get(3, [])] + [w for w, _ in idx.get(4, [])] \
        + [w for w, _ in idx.get(5, [])] + [w for w, _ in idx.get(6, [])] + [w for w, _ in idx.get(7, [])]
    l7 = ref.generate_crossword("general", words7, 7)
    r7 = score(l7, Spec(size=7, require_symmetry=True), words7)
    print("\n7x7 best-effort:\n" + render(l7) + f"\n-> valid={r7['valid']} score={r7['combined_score']} density={r7['fill_density']} reasons={r7['reasons'][:2]}")

    print(f"\nAll {PASS} reference-generator checks passed.")


if __name__ == "__main__":
    main()
