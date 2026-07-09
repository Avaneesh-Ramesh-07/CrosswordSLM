"""Strip provenance clutter from effective_spec, in place.

The specs were generated with a `topic`, `difficulty`, and `heuristic_hints` that were
used only to STEER data generation -- they are never in the model's input or the code.
On inspection they're misleading (varied topics, hint lists) and contradict the
vocabulary-only / hints-in-weights intent. This normalizes `topic` -> "vocabulary" and
removes `heuristic_hints` + `difficulty` from every record's effective_spec. Safe:
scoring/verification reads only size, require_symmetry, min_word_len, time_budget_s,
density_target. Run on the per-section source dirs, then re-run merge_dataset.

    python pipeline/clean_meta.py runs/bulk/dataset runs/gen3/dataset \
        runs/templates15/dataset runs/templates11/dataset
"""

from __future__ import annotations

import glob
import json
import os
import sys

SFT_NAMES = ("train.jsonl", "dev.jsonl", "eval.jsonl", "test.jsonl")
NEG_NAMES = ("negatives.jsonl", "negatives_eval.jsonl")


def _clean(eff):
    if not isinstance(eff, dict):
        return eff
    eff = dict(eff)
    eff["topic"] = "vocabulary"
    eff.pop("heuristic_hints", None)
    eff.pop("difficulty", None)
    return eff


def _file(path):
    name = os.path.basename(path)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        if name in SFT_NAMES and "meta" in r and "effective_spec" in r["meta"]:
            r["meta"]["effective_spec"] = _clean(r["meta"]["effective_spec"])
            n += 1
        elif name in NEG_NAMES and "effective_spec" in r:
            r["effective_spec"] = _clean(r["effective_spec"])
            n += 1
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return n


def main(dirs):
    total = 0
    for d in dirs:
        for name in SFT_NAMES + NEG_NAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                c = _file(p)
                total += c
                print(f"  {p}: cleaned {c}")
    print(f"\ncleaned effective_spec in {total} records")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("runs/*/dataset")))
