"""Spec generation: the SPEC is the model's input (and OpenEvolve's problem).

A SpecRecord is a structured description of one crossword-generation task; it
renders to a natural-language SPEC (the user turn the model learns to map to
code) and maps to a harness `Spec` for scoring. We sample specs stratified
across the axes that make the dataset diverse — size, symmetry, density target,
time budget, topic, difficulty, and optional heuristic hints (the legitimate
lever that lets one spec-region map to different valid program families).

Hard constraints in the rendered spec (size, symmetry, min length) are exactly
what the scorer enforces, so a program that ignores them fails `valid` and never
becomes a training target.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

# Allow running this file directly (python pipeline/spec_generator.py) as well as
# importing it as pipeline.spec_generator.
if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from harness.scorer import MIN_WORD_LEN, Spec

# Start with achievable sizes; 15 is the target frontier OpenEvolve/distillation
# must push toward (the reference generator won't fill it reliably yet).
SIZES = [5, 7, 9, 11, 13, 15]

# Per-size targets (white-square fraction) and time budgets, roughly NYT-shaped.
DENSITY_TARGET = {5: 0.90, 7: 0.80, 9: 0.76, 11: 0.74, 13: 0.72, 15: 0.72}
TIME_BUDGET_S = {5: 2, 7: 3, 9: 4, 11: 5, 13: 7, 15: 10}

TOPICS = [
    "general vocabulary", "SAT vocabulary", "high-school vocabulary", "science",
    "world history", "literature", "geography", "everyday words",
]
DIFFICULTY = ["easy", "medium", "hard"]

# The diversity lever: naming a technique lets the same spec map to a different
# valid program family (greedy vs. AC-3 vs. beam), which fights memorization.
HINT_POOL = [
    "MRV (minimum-remaining-values) slot ordering",
    "AC-3 arc-consistency propagation",
    "forward checking after each placement",
    "least-constraining-value ordering",
    "bitset letter domains for fast intersection",
    "a pattern index keyed by (length, position, letter)",
    "greedy placement with random restarts",
    "maintaining arc consistency (MAC) during search",
]

_SIGNATURE_BLOCK = (
    "Write a single self-contained Python function:\n"
    "    generate_crossword(topic: str, word_source: list[str], size: int) -> dict\n"
    "It must construct and fill the grid and return a layout dict of the form:\n"
    '    {"rows": int, "cols": int,\n'
    '     "cells": [{"r": int, "c": int, "letter": str, "number": int (optional)}],\n'
    '     "across": [{"number", "row", "col", "answer", "len"}], "down": [ ...same... ]}'
)


@dataclass
class SpecRecord:
    spec_id: str
    size: int
    require_symmetry: bool
    min_word_len: int
    time_budget_s: float
    density_target: float
    topic: str
    difficulty: str
    heuristic_hints: list = field(default_factory=list)
    split: str = "train"

    def to_scorer_spec(self, topic_words=()) -> Spec:
        """Map to a harness Spec for scoring. `topic_words` = the vocabulary the
        external word source chose for this topic (drives the coverage metric)."""
        return Spec(
            size=self.size,
            topic_words=tuple(topic_words),
            require_symmetry=self.require_symmetry,
            min_word_len=self.min_word_len,
            time_budget_s=self.time_budget_s,
            density_target=self.density_target,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def render_spec(rec: SpecRecord) -> str:
    """Render a SpecRecord to the natural-language SPEC used as the model input."""
    rules = [
        f"the grid is exactly {rec.size} x {rec.size}",
        f"every white run (across and down) is at least {rec.min_word_len} letters, and "
        "every white cell is checked (part of both an across and a down entry)",
        "every entry is a word drawn from word_source, never invented or hardcoded",
        "all white cells form a single connected region",
    ]
    if rec.require_symmetry:
        rules.append("the black squares are placed with 180-degree rotational symmetry")
    rules_text = "".join(f"\n  - {r};" for r in rules)
    hint = ""
    if rec.heuristic_hints:
        hint = "\nConsider techniques such as " + ", ".join(rec.heuristic_hints) + "."
    return (
        f"Task ({rec.difficulty}): generate a {rec.size}x{rec.size} fixed-grid, "
        f'American-style crossword on the topic "{rec.topic}".\n'
        f"{_SIGNATURE_BLOCK}\n"
        f"Hard rules:{rules_text}\n"
        f"Aim for a white-square density of at least {rec.density_target:.2f} and finish "
        f"within {rec.time_budget_s:g} seconds.{hint}\n"
        f"Output only the Python code."
    )


def sample_spec(rng: random.Random, spec_id: str, size: int) -> SpecRecord:
    require_symmetry = size <= 5 or rng.random() > 0.15  # mostly symmetric
    difficulty = rng.choice(DIFFICULTY)
    density = round(DENSITY_TARGET[size] + rng.choice([-0.02, 0.0, 0.0, 0.02]), 2)
    n_hints = rng.choice([0, 1, 1, 2, 2, 3])
    hints = rng.sample(HINT_POOL, k=n_hints)
    return SpecRecord(
        spec_id=spec_id,
        size=size,
        require_symmetry=require_symmetry,
        min_word_len=MIN_WORD_LEN,
        time_budget_s=TIME_BUDGET_S[size],
        density_target=density,
        topic=rng.choice(TOPICS),
        difficulty=difficulty,
        heuristic_hints=hints,
    )


def generate_specs(n: int, seed: int = 0, sizes=SIZES, dev_frac=0.1, test_frac=0.1) -> list:
    """Return `n` SpecRecords, stratified round-robin over `sizes`, with a
    per-record train/dev/test split (stratified so every size appears in each).

    Held-out program-families and held-out word-sources are applied later, in
    the harvest/build step; this split covers held-out SPECS.
    """
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        size = sizes[i % len(sizes)]
        rec = sample_spec(rng, f"s{i:05d}", size)
        r = rng.random()
        rec.split = "dev" if r < dev_frac else ("test" if r < dev_frac + test_frac else "train")
        specs.append(rec)
    return specs


def save_specs(specs, path):
    """Write a spec catalog to JSONL so the OpenEvolve run and the harvest step
    agree on spec_id -> spec."""
    with open(path, "w", encoding="utf-8") as fh:
        for s in specs:
            fh.write(json.dumps(s.as_dict()) + "\n")


def load_specs(path) -> dict:
    """Load a spec catalog written by save_specs. Returns {spec_id: SpecRecord}."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = SpecRecord(**json.loads(line))
            out[rec.spec_id] = rec
    return out


if __name__ == "__main__":
    specs = generate_specs(12, seed=0)
    from collections import Counter

    print("split counts:", dict(Counter(s.split for s in specs)))
    print("size counts :", dict(Counter(s.size for s in specs)))
    print("\n--- two sample rendered specs ---")
    for rec in (specs[0], specs[7]):
        print(f"\n[{rec.spec_id}] size={rec.size} sym={rec.require_symmetry} split={rec.split}")
        print(render_spec(rec))
