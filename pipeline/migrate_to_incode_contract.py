"""Migrate existing SFT files to the in-code-contract format, in place.

Rewrites each SFT record's `system` turn to the minimal system prompt and moves the
task contract into a COMMENT HEADER at the top of the assistant program (via
build_dataset.assistant_content). Idempotent. Run on the per-section source dirs, then
re-run merge_dataset.py to regenerate data/sft.

    python pipeline/migrate_to_incode_contract.py runs/bulk/dataset runs/gen3/dataset \
        runs/templates15/dataset runs/templates11/dataset
"""

from __future__ import annotations

import glob
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_dataset import SYSTEM, assistant_content

SFT_NAMES = ("train.jsonl", "dev.jsonl", "eval.jsonl", "test.jsonl")


def _unwrap_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s
        s = s.rstrip()
        if s.endswith("```"):
            s = s[:-3]
    return s.rstrip("\n")


def _migrate_file(path: str) -> int:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = 0
    for r in rows:
        msgs = r.get("messages")
        if not msgs or len(msgs) < 3:
            continue
        msgs[0]["content"] = SYSTEM                                  # minimal system
        inner = _unwrap_fence(msgs[2]["content"])                   # raw program (maybe already headed)
        msgs[2]["content"] = assistant_content(inner)               # contract-comment + program
        n += 1
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return n


def main(dirs):
    total = 0
    for d in dirs:
        for name in SFT_NAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                c = _migrate_file(p)
                total += c
                print(f"  {p}: migrated {c}")
    print(f"\nmigrated {total} SFT records to in-code-contract format")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("runs/*/dataset"))
    main(args)
