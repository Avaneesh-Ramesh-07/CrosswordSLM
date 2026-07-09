"""Negatives are KEPT + labeled + persisted (the SOAR "bad example" requirement).
Run: python tests/test_negatives_persist.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_dataset import write_negatives  # noqa: E402
from pipeline.harvest import SYMMETRY_REASON, classify_failure, process_harvest  # noqa: E402
from pipeline.spec_generator import SpecRecord  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def spec(sid, split="train"):
    return SpecRecord(spec_id=sid, size=7, require_symmetry=True, min_word_len=3,
                      time_budget_s=5, density_target=0.75, topic="science",
                      difficulty="medium", heuristic_hints=["AC-3 arc-consistency propagation"], split=split)


def met(valid, **kw):
    m = {"valid": valid, "combined_score": 0.2, "filler_fraction": 0.0,
         "invalid_crossing_frac": 0.0, "invalid_entry_frac": 0.0, "runtime_s": 2.0}
    m.update(kw)
    return m


def bd(**kw):
    base = {"valid": 0, "symmetry_ok": True, "connected": 1, "fill_density": 0.0, "coverage": 0.0, "reasons": []}
    base.update(kw)
    return base


CRASH = "def generate_crossword(topic, word_source, size):\n    raise ValueError('boom')\n"
EMPTY = "def generate_crossword(topic, word_source, size):\n    return {'rows': size, 'cols': size, 'cells': [], 'across': [], 'down': []}\n"
DISC = "def generate_crossword(topic, word_source, size):\n    return {'rows': size, 'disc': 1}\n"
CLEAN = "def generate_crossword(topic, word_source, size):\n    return {'rows': size, 'ok': 1}\n"


def main():
    specs = {"sA": spec("sA"), "sB": spec("sB", "dev")}
    harvest = [
        # a crashed run reports its status as a bare token (reasons == [status])
        {"spec_id": "sA", "code": CRASH, "metrics": met(0.0),
         "artifacts": {"best_draw": bd(reasons=["exception"])}},
        {"spec_id": "sA", "code": EMPTY, "metrics": met(0.0),
         "artifacts": {"best_draw": bd(reasons=["empty grid"])}},
        {"spec_id": "sB", "code": CLEAN, "metrics": met(1.0),
         "artifacts": {"best_draw": bd(valid=1, fill_density=0.80)}},
    ]
    # a repeated broken program (same code -> same hash) to exercise per_negative_cap
    harvest += [{"spec_id": "sA", "code": DISC, "metrics": met(0.0),
                 "artifacts": {"best_draw": bd(reasons=["white cells not connected"])}} for _ in range(15)]

    out = process_harvest(harvest, specs, per_negative_cap=3)
    print("failure_counts:", out["failure_counts"], "| n_negatives:", out["n_negatives"])

    cats = {n["failure_category"] for n in out["negatives"]}
    check("crash labeled exception", classify_failure(["exception"]) == "exception")
    check("substring 'oom' in 'boom' is NOT mislabeled oom", classify_failure(["raised boom error"]) != "oom")
    check("empty labeled empty_grid", "empty_grid" in cats)
    check("disconnected labeled disconnected", "disconnected" in cats)
    check("per_negative_cap collapses repeated broken program (3 not 15)",
          out["failure_counts"].get("disconnected") == 3, str(out["failure_counts"]))
    check("clean row is a solution, not a negative", out["kind_counts"].get("solution") == 1)

    tmp = tempfile.mkdtemp()
    n = write_negatives(out["negatives"], tmp)   # -> {"pool":..., "eval":...}
    path = os.path.join(tmp, "negatives.jsonl")
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    # all specs here are train/dev -> everything lands in the DPO pool, none held out
    check("negatives.jsonl written (DPO pool)", os.path.exists(path) and len(rows) == n["pool"] == out["n_negatives"])
    check("no eval negatives in this fixture", n["eval"] == 0 and os.path.exists(os.path.join(tmp, "negatives_eval.jsonl")))
    fields = {"spec_id", "spec", "effective_spec", "code", "kind", "split",
              "program_hash", "metrics", "reasons", "failure_category"}
    check("every negative record has all fields", all(fields <= set(r) for r in rows))
    check("negatives carry metrics incl filler_fraction",
          all("filler_fraction" in r["metrics"] for r in rows))
    check("kind == negative", all(r["kind"] == "negative" for r in rows))

    print(f"\nAll {PASS} negatives-persistence checks passed.")


if __name__ == "__main__":
    main()
