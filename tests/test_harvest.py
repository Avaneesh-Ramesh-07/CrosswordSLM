"""Tests for the harvest processor + dataset builder (pure Python, no sandbox).
Run: python tests/test_harvest.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_dataset import build, to_chat  # noqa: E402
from pipeline.harvest import SYMMETRY_REASON, ast_hash, process_harvest  # noqa: E402
from pipeline.spec_generator import SpecRecord, render_spec  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def spec(sid, size, sym, split):
    return SpecRecord(spec_id=sid, size=size, require_symmetry=sym, min_word_len=3,
                      time_budget_s=5, density_target=0.75, topic="SAT vocabulary",
                      difficulty="medium", heuristic_hints=["AC-3 arc-consistency propagation"], split=split)


CODE1 = "def generate_crossword(topic, word_source, size):\n    return {'rows': size}\n"
CODE2 = "def generate_crossword(topic, word_source, size):\n    x = 1\n    return {'rows': size, 'x': x}\n"
CODE3 = "def generate_crossword(topic, word_source, size):\n    return {'rows': size, 'y': 2}\n"
CODE4 = "def generate_crossword(topic, word_source, size):\n    raise ValueError\n"


def bd(**kw):
    base = {"valid": 0, "symmetry_ok": True, "connected": 1, "fill_density": 0.0, "coverage": 0.0, "reasons": []}
    base.update(kw)
    return base


def main():
    specs = {
        "sA": spec("sA", 7, True, "train"),
        "sB": spec("sB", 9, True, "dev"),
        "sC": spec("sC", 7, True, "train"),
    }
    def met(valid, combined, filler=0.0, runtime=2.0):
        return {"valid": valid, "combined_score": combined, "filler_fraction": filler,
                "invalid_crossing_frac": 0.0, "invalid_entry_frac": 0.0, "runtime_s": runtime}

    harvest = [
        # valid, clean, fast, within budget -> SOLUTION (combined no longer gates)
        {"spec_id": "sA", "code": CODE1, "metrics": met(1.0, 0.60),
         "artifacts": {"best_draw": bd(valid=1, fill_density=0.80, reasons=[])}},
        # valid but filler 0.5 > 0.30 -> misses quality bar -> HINDSIGHT_DENSITY
        {"spec_id": "sB", "code": CODE2, "metrics": met(1.0, 0.90, filler=0.5),
         "artifacts": {"best_draw": bd(valid=1, fill_density=0.70, reasons=[])}},
        # invalid, symmetry the only failing check -> HINDSIGHT_SYMMETRY
        {"spec_id": "sC", "code": CODE3, "metrics": met(0.0, 0.40),
         "artifacts": {"best_draw": bd(valid=0, symmetry_ok=False, fill_density=0.72, reasons=[SYMMETRY_REASON])}},
        # invalid, crossing conflict -> NEGATIVE (kept + labeled)
        {"spec_id": "sA", "code": CODE4, "metrics": met(0.0, 0.20),
         "artifacts": {"best_draw": bd(valid=0, reasons=["crossing letter conflict"])}},
        # dup of CODE1 solution -> capped out by per_program_cap=1
        {"spec_id": "sC", "code": CODE1, "metrics": met(1.0, 0.60),
         "artifacts": {"best_draw": bd(valid=1, fill_density=0.80, reasons=[])}},
    ]

    out = process_harvest(harvest, specs, per_program_cap=1)
    print("kind_counts:", out["kind_counts"], "| negatives:", out["n_negatives"],
          "| failures:", out["failure_counts"])

    check("3 solutions (dup capped out)", out["n_solutions"] == 3, str(out["n_solutions"]))
    check("one plain solution (metric-based, not combined)", out["kind_counts"].get("solution") == 1, str(out["kind_counts"]))
    check("one density hindsight (filler>0.30)", out["kind_counts"].get("hindsight_density") == 1)
    check("one symmetry hindsight", out["kind_counts"].get("hindsight_symmetry") == 1)
    check("one negative (conflict)", out["n_negatives"] == 1 and "crossing letter conflict" in out["negatives"][0]["reasons"][0])
    check("negative is labeled crossing_conflict", out["negatives"][0]["failure_category"] == "crossing_conflict", out["negatives"][0]["failure_category"])
    check("negative carries effective_spec + metrics", "effective_spec" in out["negatives"][0] and "filler_fraction" in out["negatives"][0]["metrics"])
    check("solutions carry effective_spec", all("effective_spec" in s for s in out["solutions"]))
    check("dedup: CODE1 counted once", sum(1 for s in out["solutions"] if s["program_hash"] == ast_hash(CODE1)) == 1)

    sym_row = next(s for s in out["solutions"] if s["kind"] == "hindsight_symmetry")
    check("symmetry-relabel drops the symmetry rule", "180-degree rotational symmetry" not in sym_row["spec"])
    dens_row = next(s for s in out["solutions"] if s["kind"] == "hindsight_density")
    check("density-relabel uses achieved density (0.70)", "at least 0.70" in dens_row["spec"], dens_row["spec"])

    # --- build_dataset: chat JSONL + splits ---
    chat = to_chat(out["solutions"][0])
    roles = [m["role"] for m in chat["messages"]]
    check("chat has system/user/assistant", roles == ["system", "user", "assistant"])
    check("assistant turn is fenced code", chat["messages"][2]["content"].startswith("```python"))
    check("meta carries spec_id + kind", "spec_id" in chat["meta"] and "kind" in chat["meta"])

    tmp = tempfile.mkdtemp()
    counts = build(out["solutions"], tmp)
    print("split counts:", counts)
    check("train has 2 (sA + sC)", counts["train"] == 2, str(counts))
    check("dev has 1 (sB)", counts["dev"] == 1)
    check("eval has 0 (held-out split, none here)", counts["eval"] == 0)
    train_rows = [json.loads(x) for x in open(os.path.join(tmp, "train.jsonl"), encoding="utf-8")]
    check("train.jsonl parses to chat examples", all("messages" in r for r in train_rows) and len(train_rows) == 2)

    print(f"\nAll {PASS} harvest/dataset checks passed.")


if __name__ == "__main__":
    main()
