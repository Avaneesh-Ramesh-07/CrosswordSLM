"""Condense each program's top-of-file module docstring to a single concise line of
GENERATION INFO + TECHNIQUES (e.g. "gen4 fusion: MRV + forward-checking + pattern-index,
palette-scaled."). The lengthy multi-paragraph explanation is dropped. Behavior-preserving:
only the module docstring is rewritten; all code (and function docstrings) is untouched.

Rules:
  * gen4/gen5 fusions already put the concise header on docstring line 1 -> keep that line verbatim.
  * fixed-template programs -> rebuild from their engine=/selection=/subset= metadata (also fixes a
    copy-paste bug where 11x11 headers said "15x15").
  * older seeds/fusions (reference, beam, gen1/2/3, vocab-first, csp_ac3) -> compose
    "<family>: <techniques>." from techniques the docstring actually names (contrast clauses like
    "distinct from csp_ac3 (AC-3)" are stripped first so a contrasted technique is not attributed).

    python pipeline/condense_headers.py
"""
from __future__ import annotations

import ast
import collections
import json
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import extract_code

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, "data", "sft_non_hardcoded_enhanced")
CHAT = ("train", "dev", "eval", "works_too_long")
FLAT = ("negatives", "negatives_eval")


def _concise_text(ds: str, size):
    """The one-line header for this docstring, or None to leave the docstring unchanged."""
    if not ds or not ds.strip():
        return None
    first = ds.split("\n")[0].strip()
    low1 = first.lower()
    lowfull = ds.lower()
    # 1) authored concise header already on line 1 (gen4/gen5) -> keep it verbatim
    if re.match(r"gen\d fusion\s*:", low1) and first.endswith(".") and len(first) <= 180:
        return first
    # 2) fixed-template -> rebuild from metadata (fixes the 15x15 copy-paste, keeps engine/selection)
    if "fixed-template" in lowfull or "baked-in" in lowfull:
        eng = re.search(r"engine=(\S+)", ds)
        sel = re.search(r"selection=(\S+)", ds)
        sub = re.search(r"subset=\S*?\((\d+)\)", ds)
        e = eng.group(1) if eng else "ac3_lcv"
        s = sel.group(1) if sel else "?"
        c = sub.group(1) if sub else "pre-verified"
        return f"fixed-template ({size}x{size}): {c} baked-in grids + {e} fill (selection={s})."
    # 3) compose <family>: <techniques>. -- strip contrast clauses so a contrasted technique
    #    ("distinct from csp_ac3 (AC-3)") is not wrongly attributed to this program.
    low = re.sub(r"(distinct from|unlike|rather than|as opposed to|no seed did|only added|was theme-blind)[^.:]*[.:]",
                 " ", lowfull)
    g = re.search(r"gen(\d)\s*fusion", low1)
    if g:
        fam = f"gen{g.group(1)} fusion"
    elif low1.startswith("reference"):
        fam = "reference v1"
    elif "beam search" in low1:
        fam = "beam search (family 3)" if "family 3" in lowfull else "beam search"
    elif "vocab-first" in low1 or "teacher generator" in low1:
        fam = "vocab-first seed"
    elif "seed" in low1 and ("ac-3" in low1 or "csp" in low1):
        fam = "csp_ac3 seed"
    else:
        fam = None
    if fam is None:
        return None
    famlow = fam.lower()
    techs = []

    def add(cond, name):
        if cond and name not in techs and name.lower() not in famlow:
            techs.append(name)

    add("ac-3" in low or "/mac" in low or " mac " in low or "arc consist" in low or "arc-consist" in low, "AC-3/MAC")
    add("mrv" in low or "most-constrain" in low or "most constrain" in low, "MRV")
    add(re.search(r"forward.?check", low), "forward-checking")
    add("lcv" in low or "least-constrain" in low or "least constrain" in low, "LCV")
    add("pattern index" in low or "pattern-index" in low, "pattern-index")
    add("beam" in low, "beam search")
    if re.search(r"(no|without)\s+backtrack", low):
        add(True, "no backtracking")
    elif "backtrack" in low:
        add(True, "backtracking")
    add("restart" in low, "random restarts")
    add(re.search(r"theme.?first", low), "theme-first")
    add(re.search(r"theme.?weight", low), "theme-weighted")
    add("vocab-first" in low or "vocab_first" in low, "vocab-first")
    if not techs:
        return None
    return f"{fam}: {' + '.join(techs)}."


def condense(code: str, size):
    """Return (new_code, changed). Rewrites only the module docstring to a single concise line."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False
    if not (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(getattr(tree.body[0].value, "value", None), str)):
        return code, False                     # no module docstring
    node = tree.body[0]
    ds = node.value.value
    text = _concise_text(ds, size)
    if text is None or text == ds.strip():
        return code, False
    lines = code.split("\n")
    new_line = f'"""{text}"""'
    new_code = "\n".join(lines[:node.lineno - 1] + [new_line] + lines[node.end_lineno:])
    try:
        if (ast.get_docstring(ast.parse(new_code)) or "").strip() != text.strip():
            return code, False
    except SyntaxError:
        return code, False
    return new_code, True


def _write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    changed = collections.Counter()
    saved = 0
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
            new, chg = condense(code, sz)
            if chg:
                r["messages"][2]["content"] = f"```python\n{new}\n```"
                changed[name] += 1
                saved += len(code) - len(new)
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
            new, chg = condense(code, sz) if sz else (code, False)
            if chg:
                r["code"] = new
                changed[name] += 1
                saved += len(code) - len(new)
            rows.append(r)
        _write(p, rows)
    print("records condensed:", dict(changed))
    print(f"chars removed from headers: {saved}")


if __name__ == "__main__":
    main()
