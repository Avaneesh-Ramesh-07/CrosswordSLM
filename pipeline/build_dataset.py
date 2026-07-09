"""Serialize harvested solutions into chat JSONL for QLoRA SFT.

Each example is a 3-turn conversation: a fixed SYSTEM contract (all the rules +
"you choose the algorithm"), a short natural USER request ("Create a 7x7 crossword
about science."), and the ASSISTANT program. All task knowledge is in SYSTEM so the
user turn stays minimal and the model must infer which technique to apply. Train
with response-only loss (mask system+user). `meta` carries curation info (kept out
of the trained turns); splits are written to separate files.
"""

from __future__ import annotations

import hashlib
import json
import os

# ALL the task knowledge lives in the SYSTEM turn (fixed), so the USER turn can be a
# short natural request ("create a 7x7 crossword about vocabulary") and the model
# decides which algorithm to apply. No per-spec hints -> the model chooses.
SYSTEM = (
    "You are an expert Python programmer specializing in crossword generation.\n"
    "The user will ask, in plain language, for a crossword of a given size on a given "
    "topic. Output EXACTLY ONE self-contained Python program (stdlib only) defining:\n"
    "    generate_crossword(topic: str, word_source, size: int) -> dict\n"
    "It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:\n"
    '    {"rows": int, "cols": int,\n'
    '     "cells": [{"r","c","letter","number"(optional)}],\n'
    '     "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}\n'
    "The crossword MUST satisfy: exactly size x size; black squares in 180-degree "
    "rotational symmetry; every white run (across and down) >= 3 letters; every white "
    "cell checked in BOTH directions; all white cells connected; every entry a real "
    "word taken from word_source; high white-square density; completes within a few "
    "seconds.\n"
    "word_source is provided at runtime (a list, or a {\"theme\",\"fill\"} dict of "
    "prioritized vocabulary + fill words); NEVER invent or hardcode words. YOU choose "
    "the construction and fill strategy (e.g. CSP backtracking with MRV + forward "
    "checking, AC-3 / maintained arc consistency, a (length,position,letter) pattern "
    "index, beam search, theme-first ordering to maximize vocabulary). Prefer packing "
    "vocabulary words where the crossings allow. Output only the Python code."
)

# Natural phrasings so the model generalizes over wording. The topic is ALWAYS
# "vocabulary" -- this is a vocabulary crossword generator, and the crossword content
# is topic-agnostic (word_source is the same educational palette regardless), so only
# the grid SIZE varies in the request.
_USER_TEMPLATES = [
    "Create a {s}x{s} fixed-grid (non-free-form) crossword about vocabulary.",
    "Make a {s}x{s} non-free-form vocabulary crossword.",
    "Generate a {s}x{s} fixed-grid crossword to teach vocabulary (not free-form).",
    "Build me a {s}x{s} vocabulary crossword on a fixed grid, not free-form.",
    "I need a {s}x{s} fixed-grid crossword for practicing vocabulary (non-free-form).",
]


def render_user_prompt(effective_spec: dict) -> str:
    """The short, natural user request -- ALWAYS about vocabulary, only the size
    varies. All rules live in SYSTEM; the model infers the technique."""
    size = (effective_spec or {}).get("size", 7)
    sid = str((effective_spec or {}).get("spec_id", ""))
    i = int(hashlib.md5(sid.encode()).hexdigest(), 16) % len(_USER_TEMPLATES)
    return _USER_TEMPLATES[i].format(s=size)


def to_chat(row: dict) -> dict:
    """One solution row -> a chat example (messages + curation meta)."""
    code = row["code"].strip()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": render_user_prompt(row.get("effective_spec"))},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ],
        "meta": {
            "spec_id": row.get("spec_id"),
            "kind": row.get("kind"),
            "combined_score": row.get("combined_score"),
            "program_hash": row.get("program_hash"),
            # effective (possibly-relaxed) spec so re-verification checks the SAME
            # constraints the example was trained on (not the original catalog spec)
            "effective_spec": row.get("effective_spec"),
            "split": row.get("split"),
        },
    }


def build(solutions, out_dir: str) -> dict:
    """Write train/dev/eval JSONL from solution rows. Returns counts per split.

    train = SFT corpus (trained). dev = validation/early-stopping (not trained).
    eval = PRISTINE held-out (never trained, never tuned) for base-vs-tuned.
    """
    os.makedirs(out_dir, exist_ok=True)
    by_split: dict = {"train": [], "dev": [], "eval": []}
    for row in solutions:
        by_split.setdefault(row.get("split", "train"), []).append(to_chat(row))

    counts = {}
    for split, examples in by_split.items():
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
        counts[split] = len(examples)
    return counts


def write_negatives(negatives, out_dir: str) -> dict:
    """Persist labeled negative (bad) examples. SFT (build) stays solutions-only;
    negatives are kept for analysis and optional preference (DPO) training.

    Eval-split negatives are written SEPARATELY (negatives_eval.jsonl) and excluded
    from the DPO pool (negatives.jsonl), so nothing derived from the held-out eval
    specs can leak into training. Each record carries kind="negative",
    failure_category, reasons, metrics, effective_spec, split, and program_hash.
    """
    os.makedirs(out_dir, exist_ok=True)
    for r in negatives:  # store the same minimal user prompt as the SFT rows (for DPO)
        r["spec"] = render_user_prompt(r.get("effective_spec"))
    pool = [r for r in negatives if r.get("split") != "eval"]      # DPO pool (train+dev)
    held = [r for r in negatives if r.get("split") == "eval"]      # held out
    for name, rows in (("negatives.jsonl", pool), ("negatives_eval.jsonl", held)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return {"pool": len(pool), "eval": len(held)}
