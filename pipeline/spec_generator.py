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
# Big grids legitimately take several seconds to CSP-fill; budgets are generous
# enough that the sandbox timeout (>= 2x budget) never kills a working generator.
TIME_BUDGET_S = {5: 2, 7: 3, 9: 5, 11: 12, 13: 20, 15: 30}

TOPICS = [
    "general vocabulary", "SAT vocabulary", "high-school vocabulary", "science",
    "world history", "literature", "geography", "everyday words",
]
DIFFICULTY = ["easy", "medium", "hard"]

# The diversity lever: naming a technique lets the same spec map to a different
# valid program family (greedy vs. AC-3 vs. beam), which fights memorization.
# TAG_TO_HINT maps scorecard heuristic TAGS to the rendered phrase, so learnings
# ("AC3 works best") can up-weight the matching hint in future generations.
TAG_TO_HINT = {
    "MRV": "MRV (minimum-remaining-values) slot ordering",
    "AC3": "AC-3 arc-consistency propagation",
    "MAC": "maintaining arc consistency (MAC) during search",
    "forward_check": "forward checking after each placement",
    "LCV": "least-constraining-value ordering",
    "bitset": "bitset letter domains for fast intersection",
    "pattern_index": "a pattern index keyed by (length, position, letter)",
    "greedy_restart": "greedy placement with random restarts",
    "template": "a pre-verified grid-template library",
    "theme_first": "theme-first ordering to seat target vocabulary in long slots",
    "two_phase": "two-phase filling: guarantee validity first, then chase coverage",
    "subdeadline_restart": "short per-attempt budgets with random restarts",
    "beam": "beam search over partial fills",
}
HINT_POOL = list(TAG_TO_HINT.values())

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


def load_hint_weights(scorecard_path) -> dict:
    """Map a generation scorecard's per-heuristic composites to per-HINT_POOL-phrase
    weights, so hints that produced good fillers are sampled more often next time.
    Phrases with no signal keep weight 1.0. Weight = max(0.1, mean_composite + 1)."""
    import json
    with open(scorecard_path, encoding="utf-8") as fh:
        card = json.load(fh)
    weights = {phrase: 1.0 for phrase in HINT_POOL}
    for row in card.get("per_heuristic", []):
        phrase = TAG_TO_HINT.get(row["heuristic"])
        if phrase:
            weights[phrase] = max(0.1, round(row["mean_composite"] + 1.0, 3))
    return weights


def _weighted_sample(rng: random.Random, items, weights, k):
    """Weighted sampling WITHOUT replacement."""
    items, weights = list(items), list(weights)
    chosen = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        if total <= 0:
            i = rng.randrange(len(items))
        else:
            r, acc, i = rng.random() * total, 0.0, len(items) - 1
            for j, w in enumerate(weights):
                acc += w
                if r <= acc:
                    i = j
                    break
        chosen.append(items.pop(i))
        weights.pop(i)
    return chosen


def sample_spec(rng: random.Random, spec_id: str, size: int, hint_weights=None) -> SpecRecord:
    require_symmetry = size <= 5 or rng.random() > 0.15  # mostly symmetric
    difficulty = rng.choice(DIFFICULTY)
    density = round(DENSITY_TARGET[size] + rng.choice([-0.02, 0.0, 0.0, 0.02]), 2)
    n_hints = rng.choice([0, 1, 1, 2, 2, 3])
    if hint_weights:
        hints = _weighted_sample(rng, HINT_POOL, [hint_weights.get(h, 1.0) for h in HINT_POOL], n_hints)
    else:
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


def generate_specs(n: int, seed: int = 0, sizes=SIZES, dev_frac=0.1, eval_frac=0.1,
                   hint_weights=None) -> list:
    """Return `n` SpecRecords, stratified round-robin over `sizes`, with a per-record
    split over held-out SPECS:
      - train (~80%): the SFT corpus (trained on).
      - dev   (~10%): validation / early-stopping (no gradient updates, but used for
                      model selection).
      - eval  (~10%): a PRISTINE held-out set -- never trained on AND never used for
                      tuning; touched only for the final base-vs-tuned comparison.

    `hint_weights` (from load_hint_weights) biases which heuristic hints appear,
    carrying a prior generation's learnings forward. Held-out program-families and
    word-sources are additional contamination controls applied later.
    """
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        size = sizes[i % len(sizes)]
        rec = sample_spec(rng, f"s{i:05d}", size, hint_weights=hint_weights)
        r = rng.random()
        rec.split = "dev" if r < dev_frac else ("eval" if r < dev_frac + eval_frac else "train")
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
