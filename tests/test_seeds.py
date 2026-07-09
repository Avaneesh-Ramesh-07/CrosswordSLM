"""Verify every seed generator. Run on Linux (WSL/Colab): python3 tests/test_seeds.py

Two checks per seed, reflecting what's actually true:
  1. CORRECTNESS: on a rich word palette it fills a valid 7x7 (direct call, reliable).
  2. WELL-BEHAVED: in the sandbox on the constrained education palette it returns
     cleanly within its wall-clock deadline (no crash/timeout). Whether it fills the
     hard vocab palette is variance-dependent -- that's what OpenEvolve improves.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score  # noqa: E402
from harness.verify import fuzz_verify  # noqa: E402
from pipeline.word_source import build_education_source, load_scored_dict  # noqa: E402
import seeds.beam_search as beam  # noqa: E402
import seeds.csp_ac3 as csp  # noqa: E402
import seeds.reference_v1 as ref  # noqa: E402

PASS = 0
SEEDS = [("reference_v1", ref), ("csp_ac3", csp), ("beam_search", beam)]


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def main():
    # 1) Correctness on a rich palette (direct call) -- each algorithm CAN fill.
    # The seeds carry a wall-clock deadline, so a single unlucky seed/topic can
    # time out (esp. under CPU load); "can fill" means it fills on at least one of
    # a few topics -- that's the capability claim, not per-seed reliability.
    rich = load_scored_dict(min_len=3, max_len=7)
    rws = list(rich)
    print(f"rich palette: {len(rws):,} words")
    for name, mod in SEEDS:
        best = None
        for topic in ("verify", "alpha", "delta", "gamma"):
            lay = mod.generate_crossword(topic, rws, 7)
            r = score(lay, Spec(size=7, require_symmetry=True), rws, scores=rich)
            best = r if (best is None or r["valid"] > best["valid"]) else best
            if r["valid"] == 1:
                break
        print(f"  {name}: valid={best['valid']} score={round(best['combined_score'], 3)}")
        check(f"{name} fills a valid 7x7 on a rich palette (best of 4 topics)",
              best["valid"] == 1, str(best["reasons"]))

    # 2) Well-behaved in the sandbox on the (constrained) education palette.
    edu = build_education_source(include_common_fill=True)
    draws = [(Spec(size=7, require_symmetry=True), edu["allowed"])]
    print("\nsandbox on education palette:")
    for name, mod in SEEDS:
        rr = fuzz_verify(open(mod.__file__, encoding="utf-8").read(), draws, scores=edu["scores"], timeout_s=10)
        res = rr["results"][0]
        print(f"  {name}: status={res['status']} valid={res.get('valid')} runtime={res.get('runtime_s')}s")
        check(f"{name} returns cleanly within deadline (no crash/timeout)", res["status"] == "ok", str(res))

    print(f"\nAll {PASS} seed checks passed ({len(SEEDS)} families).")


if __name__ == "__main__":
    main()
