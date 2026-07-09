"""Build the GRPO prompt set from the RLVR dataset snapshot (rlvr/dataset/).

Construct-from-scratch prompts only (meta.kind != 'fixed_template'), sizes {7,9}
(the sizes that fill reliably from scratch; 11/15 are the fixed-template task).
The SFT user turns collapse to ~10 unique (system,user) strings (5 phrasings x 2
sizes) -- that low diversity IS the deployment distribution, so compensate with
num_generations / temperature / n_repeats at train time, NOT synthetic prompts.
dev/eval prompts are held out for the SFT-vs-RLVR comparison.

Each record is a conversational GRPO row: {"prompt": [system, user], "size": int}.
TRL forwards the non-"prompt" columns (here `size`) to the reward function, which
derives the canonical per-size Spec from it.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TRAIN = os.path.join(_ROOT, "rlvr", "dataset", "train.jsonl")
_DEFAULT_EVAL = os.path.join(_ROOT, "rlvr", "dataset", "eval.jsonl")
CONSTRUCT_SIZES = (7, 9)


def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _size_of(row: dict):
    return ((row.get("meta") or {}).get("effective_spec") or {}).get("size")


def is_construct(row: dict) -> bool:
    return (row.get("meta") or {}).get("kind") != "fixed_template"


def _prompt_messages(row: dict) -> list:
    """Keep only system + user turns (drop the assistant target)."""
    return [m for m in row["messages"] if m.get("role") in ("system", "user")]


def _dedup_records(path: str, sizes) -> list:
    seen: dict = {}
    for row in load_rows(path):
        if not is_construct(row):
            continue
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


def canonical_prompt_records(train_path: str = _DEFAULT_TRAIN, sizes=CONSTRUCT_SIZES) -> list:
    """Unique construct-from-scratch (system,user) prompts, one record each."""
    return _dedup_records(train_path, sizes)


def held_out_eval_prompts(eval_path: str = _DEFAULT_EVAL, sizes=CONSTRUCT_SIZES) -> list:
    """Held-out eval prompts (never used for GRPO training)."""
    return _dedup_records(eval_path, sizes)


def build_grpo_dataset(train_path: str = _DEFAULT_TRAIN, sizes=CONSTRUCT_SIZES, n_repeats: int = 1):
    """A datasets.Dataset of {prompt, size} rows for TRL GRPOTrainer.

    `n_repeats` duplicates the (few) unique prompts so an epoch has more optimizer
    steps; real diversity still comes from num_generations/temperature at rollout.
    """
    from datasets import Dataset  # lazy: not needed for local dryrun
    recs = canonical_prompt_records(train_path, sizes) * int(n_repeats)
    return Dataset.from_list(recs)


if __name__ == "__main__":
    recs = canonical_prompt_records()
    print(f"{len(recs)} unique construct prompts (sizes {CONSTRUCT_SIZES}); "
          f"by size: {dict(Counter(r['size'] for r in recs))}")
    for r in recs[:3]:
        print("---")
        for m in r["prompt"]:
            print(f"[{m['role']}] {m['content'][:110]}")
