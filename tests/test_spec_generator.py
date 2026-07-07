"""Tests for the spec generator. Run: python tests/test_spec_generator.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.spec_generator import SIZES, generate_specs, render_spec, sample_spec  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def test_count_and_determinism():
    a = generate_specs(60, seed=0)
    b = generate_specs(60, seed=0)
    check("count == 60", len(a) == 60)
    check("deterministic ids", [s.spec_id for s in a] == [s.spec_id for s in b])
    check("deterministic render", [render_spec(s) for s in a] == [render_spec(s) for s in b])
    c = generate_specs(60, seed=1)
    check("different seed -> different specs", [render_spec(s) for s in a] != [render_spec(s) for s in c])


def test_stratification_and_splits():
    specs = generate_specs(120, seed=0)
    sizes = {s.size for s in specs}
    check("all sizes represented", sizes == set(SIZES), str(sorted(sizes)))
    splits = {s.split for s in specs}
    check("splits are train/dev/test only", splits <= {"train", "dev", "test"}, str(splits))
    check("train present", "train" in splits)
    check("dev present", "dev" in splits)
    check("test present", "test" in splits)


def test_render_contents():
    specs = generate_specs(60, seed=2)
    for s in specs:
        text = render_spec(s)
        assert f"{s.size} x {s.size}" in text, f"size missing in {s.spec_id}"
        assert "word_source" in text
        assert "connected" in text
        assert "generate_crossword(" in text
        assert text.strip().endswith("Output only the Python code.")
        has_sym = "180-degree rotational symmetry" in text
        assert has_sym == s.require_symmetry, f"symmetry mismatch {s.spec_id}"
    check("render includes size/word_source/connected/signature for all", True)
    check("symmetry rule appears iff required", True)


def test_to_scorer_spec():
    rec = generate_specs(10, seed=3)[0]
    spec = rec.to_scorer_spec(topic_words=["COMET", "ORBIT"])
    check("maps size", spec.size == rec.size)
    check("maps symmetry", spec.require_symmetry == rec.require_symmetry)
    check("maps time budget", spec.time_budget_s == rec.time_budget_s)
    check("maps density target", spec.density_target == rec.density_target)
    check("passes topic words", spec.topic_words == ("COMET", "ORBIT"))


if __name__ == "__main__":
    tests = [test_count_and_determinism, test_stratification_and_splits, test_render_contents, test_to_scorer_spec]
    for t in tests:
        t()
    print(f"\nAll {PASS} spec-generator checks passed across {len(tests)} tests.")
