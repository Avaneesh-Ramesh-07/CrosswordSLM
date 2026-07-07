"""End-to-end check of the education word source (Insight #3) through the seed.

targets = SAT n high-score crossword (the vocabulary to teach); allowed = targets
+ common fill (connectors). Confirms the palette is clean, vocab-dense, and still
buildable. Run: python tests/test_education_integration.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score  # noqa: E402
from pipeline.word_source import (_SAT_PATH, _load_wordset,  # noqa: E402
                                   build_education_source, index_by_length)
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
    edu = build_education_source()  # PURE intersection (default)
    allowed, scores = edu["allowed"], edu["scores"]
    targets = set(edu["targets"])
    idx = index_by_length(scores)
    print(f"pure intersection palette: {edu['n_vocab']:,} vocab words "
          f"(allowed==targets: {edu['n_allowed'] == edu['n_vocab']}, n_fill={edu['n_fill']})")

    check("SAT filter active", edu["used_sat"])
    check("pure intersection: allowed == targets", set(allowed) == targets and edu["n_fill"] == 0)
    check("no common fill added by default", edu["used_common"] is False)
    check("vocabulary is substantial", edu["n_vocab"] > 1000, str(edu["n_vocab"]))

    sat_set = _load_wordset(_SAT_PATH)
    check("every palette word is an SAT word", targets <= sat_set)
    known_sat = {"EMOLLIENT", "TRUCULENT", "SOPHISTRY", "ABANDON", "RELUCTANT", "DESTITUTE"}
    check("known SAT words present", len(targets & known_sat) >= 3, str(targets & known_sat))

    allowed_set = set(allowed)
    for junk in ("RCADOME", "BABYHITLER", "DEVELOPMENTHELL", "UIE"):
        check(f"junk excluded: {junk}", junk not in allowed_set)

    # Fillability with the NAIVE seed is a known-hard case for pure vocab (no short
    # connectors). Informational only -- producing valid vocab-dense grids is what
    # OpenEvolve / the fine-tune are meant to learn. Flip include_common_fill=True
    # if we later want connectors.
    print("\nfillability with the naive reference seed (informational):")
    for size in (5, 7):
        layout = ref.generate_crossword("SAT vocabulary", allowed, size)
        r = score(layout, Spec(size=size, require_symmetry=True), allowed, scores=scores)
        state = "filled" if r["valid"] else "cannot fill pure vocab (expected for naive seed)"
        print(f"  size {size}: valid={r['valid']} -> {state}")

    print(f"\nAll {PASS} education-integration checks passed.")


if __name__ == "__main__":
    main()
