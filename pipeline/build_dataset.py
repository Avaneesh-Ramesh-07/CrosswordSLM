"""Serialize harvested (spec -> program) solutions into chat JSONL for QLoRA SFT.

Each example is a 3-turn conversation (system contract, user SPEC, assistant
program). Train with response-only loss (mask the system+user) so the model
learns to PRODUCE the program, not memorize specs. Splits are written to
separate files; `meta` carries curation info (kept out of the trained turns).
"""

from __future__ import annotations

import json
import os

SYSTEM = (
    "You are an expert Python programmer specializing in crossword generation. "
    "Given a SPEC, output exactly ONE self-contained Python program defining "
    "generate_crossword(topic, word_source, size) that returns the described "
    "fixed-grid crossword layout. The word list is provided as word_source; never "
    "invent or hardcode words. Output only the Python code."
)


def to_chat(row: dict) -> dict:
    """One solution row -> a chat example (messages + curation meta)."""
    code = row["code"].strip()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["spec"]},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ],
        "meta": {
            "spec_id": row.get("spec_id"),
            "kind": row.get("kind"),
            "combined_score": row.get("combined_score"),
            "program_hash": row.get("program_hash"),
        },
    }


def build(solutions, out_dir: str) -> dict:
    """Write train/dev/test JSONL from solution rows. Returns counts per split."""
    os.makedirs(out_dir, exist_ok=True)
    by_split: dict = {"train": [], "dev": [], "test": []}
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
