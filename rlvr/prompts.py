"""Build the GRPO prompt set from the hardcoded-words SFT corpus.

The SLM being refined was trained on data/sft_hardcoded_words/ (targets are
self-contained generate_crossword(topic="vocabulary", word_source=None, size=N)
with an embedded _WORDS list). The user turns are size-routed only, so they
collapse to ~5 phrasings per size = 20 unique prompts across sizes 7/9/11/15.
That low diversity IS the deployment distribution -- compensate with
num_generations / temperature / n_repeats, NOT synthetic prompts. dev/eval held out.

Each record is a conversational GRPO row: {"prompt": [system, user], "size": int}.
TRL forwards the non-"prompt" columns (here `size`) to the reward, which derives
the canonical per-size Spec from it.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TRAIN = os.path.join(_ROOT, "data", "sft_hardcoded_words", "train.jsonl")
_DEFAULT_EVAL = os.path.join(_ROOT, "data", "sft_hardcoded_words", "eval.jsonl")
SIZES = (7, 9, 11, 15)


def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _size_of(row: dict):
    return ((row.get("meta") or {}).get("effective_spec") or {}).get("size")


def _prompt_messages(row: dict) -> list:
    """Keep only system + user turns (drop the assistant target)."""
    return [m for m in row["messages"] if m.get("role") in ("system", "user")]


def _dedup_records(path: str, sizes) -> list:
    """Unique (system,user) prompts across ALL rows for the given sizes.

    We do NOT filter by meta.kind: sizes 11/15 are entirely kind=='fixed_template'
    (template-fill) while 7/9 are from-scratch, and we want all four sizes. Deduping
    by prompt text collapses each size to its ~5 phrasings regardless of kind.
    """
    seen: dict = {}
    for row in load_rows(path):
        size = _size_of(row)
        if size not in sizes:
            continue
        prompt = _prompt_messages(row)
        sys_txt = next((m["content"] for m in prompt if m["role"] == "system"), "")
        usr_txt = next((m["content"] for m in prompt if m["role"] == "user"), "")
        key = (sys_txt, usr_txt)
        if key not in seen:
            seen[key] = {"prompt": prompt, "size": int(size)}
    return list(seen.values())


def canonical_prompt_records(train_path: str = _DEFAULT_TRAIN, sizes=SIZES) -> list:
    return _dedup_records(train_path, sizes)


def held_out_eval_prompts(eval_path: str = _DEFAULT_EVAL, sizes=SIZES) -> list:
    return _dedup_records(eval_path, sizes)


def build_grpo_dataset(train_path: str = _DEFAULT_TRAIN, sizes=SIZES, n_repeats: int = 1):
    """A datasets.Dataset of {prompt, size} rows for TRL GRPOTrainer."""
    from datasets import Dataset  # lazy: not needed for local dryrun
    recs = canonical_prompt_records(train_path, sizes) * int(n_repeats)
    return Dataset.from_list(recs)


if __name__ == "__main__":
    recs = canonical_prompt_records()
    print(f"{len(recs)} unique prompts (sizes {SIZES}); by size: "
          f"{dict(Counter(r['size'] for r in recs))}")
    for r in recs[:4]:
        print("---")
        for m in r["prompt"]:
            print(f"[{m['role']}] {m['content'][:110]}")
