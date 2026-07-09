"""Re-render user prompts in existing dataset files in place.

The user turn is generated from the embedded effective_spec (see
build_dataset.render_user_prompt). When the prompt templates change (e.g. adding
"non-free-form"), run this to re-render every SFT and negative record without
re-harvesting. SFT files rewrite messages[1] (the user turn); negative files
rewrite the "spec" field. The assistant program and all curation meta are
untouched.
"""

from __future__ import annotations

import glob
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_dataset import render_user_prompt

SFT_NAMES = ("train.jsonl", "dev.jsonl", "eval.jsonl", "test.jsonl")
NEG_NAMES = ("negatives.jsonl", "negatives_eval.jsonl")


def _rewrite(path: str) -> int:
    name = os.path.basename(path)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        if name in SFT_NAMES and "messages" in r:
            eff = (r.get("meta") or {}).get("effective_spec")
            if eff is not None:
                r["messages"][1]["content"] = render_user_prompt(eff)
                n += 1
        elif name in NEG_NAMES:
            eff = r.get("effective_spec")
            if eff is not None:
                r["spec"] = render_user_prompt(eff)
                n += 1
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return n


def main(dirs):
    total = 0
    prompts = set()
    for d in dirs:
        for name in SFT_NAMES + NEG_NAMES:
            path = os.path.join(d, name)
            if os.path.exists(path):
                c = _rewrite(path)
                total += c
                print(f"  {path}: re-rendered {c}")
    # report the distinct prompts now present
    for d in dirs:
        p = os.path.join(d, "train.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                prompts.add(json.loads(l)["messages"][1]["content"])
    print(f"\nre-rendered {total} records across {len(dirs)} dir(s)")
    print("distinct user prompts:")
    for pr in sorted(prompts):
        print("  -", pr)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        dirs = args
    else:
        dirs = sorted(glob.glob("runs/*/dataset"))
    main(dirs)
