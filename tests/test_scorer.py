"""Unit tests for harness.scorer.

Runs as a plain script (no pytest dependency): `python tests/test_scorer.py`.
Exits non-zero on the first failed assertion.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, build_layout_from_grid, score  # noqa: E402
from tests import fixtures  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(name + (f": {detail}" if detail else ""))
    PASS += 1


def valid_layout_and_words():
    g = fixtures.grid_map()
    n = fixtures.size()
    layout = build_layout_from_grid(g, n)
    words = {e["answer"] for e in layout["across"]} | {e["answer"] for e in layout["down"]}
    return layout, words, n


# 1) The real NYT grid must score fully valid and high.
def test_valid_nyt():
    layout, words, n = valid_layout_and_words()
    spec = Spec(size=n, require_symmetry=True)
    r = score(layout, spec, words)
    check("nyt: valid == 1", r["valid"] == 1, str(r["reasons"]))
    check("nyt: connected", r["connected"] == 1)
    check("nyt: symmetric", r["symmetry_ok"] is True)
    expected_density = round(len(fixtures.grid_map()) / (n * n), 4)
    check("nyt: density matches white-cell fraction", r["fill_density"] == expected_density, r["fill_density"])
    check("nyt: density >= tau (0.72)", r["fill_density"] >= 0.72, r["fill_density"])
    check("nyt: fill_quality neutral (no scores)", r["fill_quality"] == 0.5, r["fill_quality"])
    check("nyt: combined_score > 0.85", r["combined_score"] > 0.85, r["combined_score"])
    check("nyt: combined_gated == combined_score", r["combined_gated"] == r["combined_score"])
    check("nyt: no accidental runs", r["accidental"] == 0)


# 2) Missing a word from the dictionary -> not-a-real-word -> invalid, score capped.
def test_missing_word():
    layout, words, n = valid_layout_and_words()
    drop = "MANATEES"
    reduced = {w for w in words if w != drop}
    r = score(layout, Spec(size=n), reduced)
    check("missing word: valid == 0", r["valid"] == 0)
    check("missing word: flagged", any("not real words" in x for x in r["reasons"]), str(r["reasons"]))
    check("missing word: score capped < 0.6", r["combined_score"] < 0.6, r["combined_score"])
    check("missing word: gated == 0", r["combined_gated"] == 0.0)


# 3) A crossing conflict is detected independent of what entries claim.
def test_conflict():
    layout, words, n = valid_layout_and_words()
    for e in layout["across"]:
        if e["answer"] == "AHEM":  # crosses ADAM (down) at (0,0)
            e["answer"] = "XHEM"
            break
    r = score(layout, Spec(size=n), words | {"XHEM"})
    check("conflict: valid == 0", r["valid"] == 0)
    check("conflict: flagged", any("conflict" in x for x in r["reasons"]), str(r["reasons"]))


# 4) Breaking 180-degree symmetry fails (with symmetry required).
def test_symmetry():
    g = fixtures.grid_map()
    n = fixtures.size()
    g[(0, 4)] = "X"  # (0,4) was black; its rotational partner (14,10) stays black
    layout = build_layout_from_grid(g, n)
    words = {e["answer"] for e in layout["across"]} | {e["answer"] for e in layout["down"]}
    r = score(layout, Spec(size=n, require_symmetry=True), words)
    check("symmetry: valid == 0", r["valid"] == 0)
    check("symmetry: flagged", any("symmetric" in x for x in r["reasons"]), str(r["reasons"]))
    # And with symmetry NOT required, the symmetry reason must disappear.
    r2 = score(layout, Spec(size=n, require_symmetry=False), words)
    check("symmetry: not flagged when not required", not any("symmetric" in x for x in r2["reasons"]))


# 5) Coverage reflects how many topic words were placed.
def test_coverage():
    layout, words, n = valid_layout_and_words()
    topic = ("MANATEES", "PRAGMATIC_NOT_HERE", "CLOVE")  # 2 of 3 are in the grid
    r = score(layout, Spec(size=n, topic_words=topic), words)
    check("coverage: between 0 and 1", 0 < r["coverage"] < 1, r["coverage"])


# 6) fill_quality rewards high-scored answers and is dragged down by the worst.
def test_fill_quality():
    layout, words, n = valid_layout_and_words()
    spec = Spec(size=n, require_symmetry=True)

    hi = score(layout, spec, words, scores={w: 80 for w in words})
    lo = score(layout, spec, words, scores={w: 40 for w in words})
    check("fill_quality: high scores -> ~0.8", abs(hi["fill_quality"] - 0.8) < 1e-6, hi["fill_quality"])
    check("fill_quality: low scores -> ~0.4", abs(lo["fill_quality"] - 0.4) < 1e-6, lo["fill_quality"])
    check("fill_quality: better fill -> higher combined_score", hi["combined_score"] > lo["combined_score"])

    # One terrible word must drag quality below all-good, via the min term.
    worst = {w: 80 for w in words}
    worst[sorted(words)[0]] = 20
    mixed = score(layout, spec, words, scores=worst)
    check("fill_quality: worst-answer penalty", mixed["fill_quality"] < hi["fill_quality"], mixed["fill_quality"])


# 7) Garbage schema is reported, not crashed.
def test_bad_schema():
    r = score({"rows": 15, "cols": 15}, Spec(size=15), [])
    check("bad schema: status", r["status"] == "bad_schema")
    check("bad schema: valid == 0", r["valid"] == 0)


if __name__ == "__main__":
    tests = [
        test_valid_nyt,
        test_missing_word,
        test_conflict,
        test_symmetry,
        test_coverage,
        test_fill_quality,
        test_bad_schema,
    ]
    for t in tests:
        t()
    print(f"\nAll {PASS} checks passed across {len(tests)} tests.")
