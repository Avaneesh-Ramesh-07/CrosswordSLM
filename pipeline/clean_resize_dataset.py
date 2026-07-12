"""In-place transform of data/sft_non_hardcoded_enhanced/:
  (1-3) scrub 180-symmetry / OpenEvolve-seed / NYT prose from every program (docstrings/comments
        only -- behavior-preserving), and
  (4)   set each record's prompt to its SIZE-SPECIFIC contract user_contract(size), where size is
        the record's already-confirmed working size (meta.effective_spec.size). No re-verification.
data/sft/ (the backup copy) is untouched. Files are written atomically (.tmp -> os.replace).

    python pipeline/clean_resize_dataset.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_dataset import SYSTEM, assistant_content, clean_program
from pipeline.contract_prompt import user_contract
from pipeline.eval_harness import extract_code

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, "data", "sft_non_hardcoded_enhanced")
CHAT = ("train", "dev", "eval", "works_too_long")
FLAT = ("negatives", "negatives_eval")
TOKENS = ("180", "symmetr", "nyt", "openevolve", "seed for")


def _write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    rep = collections.Counter()
    for name in CHAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        rows = []
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            sz = (r.get("meta", {}).get("effective_spec") or {}).get("size")
            code = extract_code(r["messages"][2]["content"]) or ""
            r["messages"][0]["content"] = SYSTEM
            r["messages"][1]["content"] = user_contract(sz)      # size-specific prompt
            r["messages"][2]["content"] = assistant_content(code)  # scrub + re-fence
            rows.append(r)
            rep[name] += 1
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
            r["code"] = clean_program(r.get("code", ""))
            r["spec"] = user_contract(sz)
            rows.append(r)
            rep[name] += 1
        _write(p, rows)

    print("re-serialized records:", dict(rep))

    # --- verify: residual target tokens + size-matched prompts ---
    print("\nresidual target tokens in program text (expect NONE):")
    for name in CHAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        res, mism = collections.Counter(), 0
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            prog = extract_code(r["messages"][2]["content"]) or ""
            for t in TOKENS:
                if t in prog.lower():
                    res[t] += 1
            sz = r["meta"]["effective_spec"]["size"]
            if f"{sz}x{sz}" not in r["messages"][1]["content"]:
                mism += 1
        print(f"  {name}: tokens={dict(res) if res else 'NONE'} | prompt-size mismatches={mism}")
    for name in FLAT:
        p = os.path.join(D, name + ".jsonl")
        if not os.path.exists(p):
            continue
        res = collections.Counter()
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            for t in TOKENS:
                if t in (r.get("code", "") or "").lower():
                    res[t] += 1
        print(f"  {name}: tokens={dict(res) if res else 'NONE'}")


if __name__ == "__main__":
    main()
