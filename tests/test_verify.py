"""fuzz_verify tests. Run on Linux (WSL/Colab): python3 tests/test_verify.py

Uses a 'hardcoder' candidate that always returns the NYT grid regardless of
input. It should pass a draw whose word_source contains those words, but the
multi-draw gate must REJECT it once a draw uses a different word_source — which
is exactly the word-hardcoding failure fuzz-verify exists to catch.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, build_layout_from_grid  # noqa: E402
from harness.verify import fuzz_verify  # noqa: E402
from tests import fixtures  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def hardcoder_candidate():
    layout = build_layout_from_grid(fixtures.grid_map(), fixtures.size())
    return "import json\nLAYOUT = json.loads(r'''" + json.dumps(layout) + "''')\n" \
           "def generate_crossword(topic, word_source, size):\n    return LAYOUT\n"


def nyt_words():
    layout = build_layout_from_grid(fixtures.grid_map(), fixtures.size())
    return sorted({e["answer"] for e in layout["across"]} | {e["answer"] for e in layout["down"]})


TIMEOUT_CANDIDATE = """
def generate_crossword(topic, word_source, size):
    while True:
        pass
"""


def main():
    code = hardcoder_candidate()
    words = nyt_words()
    n = fixtures.size()

    # All draws use the matching word_source -> every draw valid -> accepted.
    good_draws = [(Spec(size=n, require_symmetry=True), words) for _ in range(2)]
    r = fuzz_verify(code, good_draws)
    check("hardcoder accepted when all draws match", r["accepted"] is True, str(r["n_valid"]))
    check("hardcoder n_valid == 2", r["n_valid"] == 2)
    check("mean_score high", r["mean_score"] > 0.85, str(r["mean_score"]))

    # Mixed draws: one matching, one with a different word_source -> rejected.
    mixed = [
        (Spec(size=n, require_symmetry=True), words),
        (Spec(size=n, require_symmetry=True), ["CAT", "DOG", "EMU"]),
    ]
    r = fuzz_verify(code, mixed)
    check("hardcoder REJECTED under varied word_source", r["accepted"] is False)
    check("mixed n_valid == 1", r["n_valid"] == 1, str(r["n_valid"]))

    # A timeouting program is never accepted.
    r = fuzz_verify(TIMEOUT_CANDIDATE, [(Spec(size=n), words)], timeout_s=1)
    check("timeout candidate rejected", r["accepted"] is False)
    check("timeout status recorded", r["results"][0]["status"] == "timeout", str(r["results"][0]))

    print(f"\nAll {PASS} fuzz-verify checks passed.")


if __name__ == "__main__":
    main()
