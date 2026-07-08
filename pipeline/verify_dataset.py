"""Dataset integrity check: re-run EVERY (spec -> program) example through the
harness and confirm each training target actually produces a valid crossword.

Each chat example's assistant turn is a generator program; we strip the code
fence, rebuild the scorer Spec from the spec catalog, and run it in the sandbox
against the education palette. Reports per-example valid/score and a summary.

Usage:
    python pipeline/verify_dataset.py --dataset runs/teacher_pilot/dataset/train.jsonl \
                                      --specs   runs/teacher_pilot/specs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.oe_evaluator import evaluate_code
from pipeline.spec_generator import load_specs
from pipeline.word_source import build_education_source


def extract_code(content: str) -> str:
    c = content.strip()
    if c.startswith("```"):
        c = c.split("\n", 1)[1] if "\n" in c else ""   # drop the ```python line
        if c.rstrip().endswith("```"):
            c = c.rstrip()[:-3]
    return c.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--specs", required=True)
    args = ap.parse_args()

    specs = load_specs(args.specs)
    edu = build_education_source(include_common_fill=True)
    theme, fill, scores = edu["targets"], edu["fill_words"], edu["scores"]
    word_source = {"theme": theme, "fill": fill}

    rows = [json.loads(l) for l in open(args.dataset, encoding="utf-8")]
    print(f"verifying {len(rows)} examples from {args.dataset}\n")

    n_valid = 0
    failures = []
    for i, row in enumerate(rows):
        code = extract_code(row["messages"][2]["content"])
        spec_id = row["meta"]["spec_id"]
        rec = specs.get(spec_id)
        if rec is None:
            print(f"[{i:2d}] {spec_id}: SPEC NOT FOUND"); failures.append(i); continue
        spec = rec.to_scorer_spec(topic_words=theme)
        out = evaluate_code(code, spec, word_source, scores=scores, n_draws=1)
        m = out["metrics"]
        bd = out["artifacts"]["best_draw"]
        ok = int(round(m["valid"])) == 1
        n_valid += ok
        tag = "OK " if ok else "FAIL"
        print(f"[{i:2d}] {tag} {spec_id} size={rec.size} prog={row['meta']['program_hash'][:8]} "
              f"valid={m['valid']} score={m['combined_score']} cov={m['coverage']} "
              + ("" if ok else f"reasons={bd.get('reasons')}"))
        if not ok:
            failures.append(i)

    print(f"\n{n_valid}/{len(rows)} examples produce a valid crossword.")
    if failures:
        print(f"failing indices: {failures}")


if __name__ == "__main__":
    main()
