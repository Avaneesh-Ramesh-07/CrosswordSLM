"""Apply specialize_to_size() across data/sft_non_hardcoded_enhanced/: rewrite every record's
program so it carries ONLY the template for that record's grid size (drop other-size `_TEMPLATES_<N>`
data + collapse the size dispatch). Behavior at the record's size is preserved (verified separately).

meta.program_hash is kept UNCHANGED -- it is the program's lineage id (used by wtl_keep.json and the
notebook's dedup); the specialized code is the same program restricted to its size. Files are written
atomically. data/sft/ (the original backup) is untouched.

    python pipeline/specialize_dataset.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import extract_code
from pipeline.specialize_templates import specialize_to_size, template_sizes

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, "data", "sft_non_hardcoded_enhanced")
CHAT = ("train", "dev", "eval", "works_too_long")
FLAT = ("negatives", "negatives_eval")


def _write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    changed = collections.Counter()
    skipped = collections.Counter()   # note -> count (records that carried extras but were NOT changed)
    saved_chars = 0

    for name in CHAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        rows = []
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            sz = r["meta"]["effective_spec"]["size"]
            code = extract_code(r["messages"][2]["content"]) or ""
            if template_sizes(code) - {sz}:
                new, chg, note = specialize_to_size(code, sz)
                if chg:
                    r["messages"][2]["content"] = f"```python\n{new}\n```"
                    changed[name] += 1
                    saved_chars += len(code) - len(new)
                else:
                    skipped[note] += 1
            rows.append(r)
        _write(p, rows)

    for name in FLAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        rows = []
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            sz = (r.get("effective_spec") or {}).get("size")
            code = r.get("code", "") or ""
            if sz and template_sizes(code) - {sz}:
                new, chg, note = specialize_to_size(code, sz)
                if chg:
                    r["code"] = new
                    changed[name] += 1
                    saved_chars += len(code) - len(new)
                else:
                    skipped[note] += 1
            rows.append(r)
        _write(p, rows)

    print("records specialized:", dict(changed))
    print("records with extras left unchanged:", dict(skipped) if skipped else "NONE")
    print(f"total chars removed: {saved_chars}")

    # verify: no record anywhere still carries a template for a size other than its own
    print("\nverify -- records still carrying other-size templates (expect 0):")
    for name in CHAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        bad = 0
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            sz = r["meta"]["effective_spec"]["size"]
            code = extract_code(r["messages"][2]["content"]) or ""
            if template_sizes(code) - {sz}:
                bad += 1
        print(f"  {name}: {bad}")
    for name in FLAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        bad = 0
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            sz = (r.get("effective_spec") or {}).get("size")
            code = r.get("code", "") or ""
            if sz and template_sizes(code) - {sz}:
                bad += 1
        print(f"  {name}: {bad}")


if __name__ == "__main__":
    main()
