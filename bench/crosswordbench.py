"""CrossWordBench-derived evaluation for crossword-generation models.

Eval protocol (per puzzle in data/crosswordbench/*.jsonl):
  1. Extract the puzzle's exact WORD SET, grid SIZE, and BLACK-SQUARE count.
  2. Render those as a SPEC (the model's input) using the SAME renderer the
     model was trained on (pipeline.spec_generator.render_spec), with the
     black-square count encoded as the density target and the word set passed
     as the runtime word_source. The model returns a crossword configuration
     (directly, or as a program we run in the sandbox).
  3. SUCCESS (binary) = that configuration is VALID under harness.scorer.
  4. Because pass/fail is coarse, also report CROSSINGS (word intersections),
     word COVERAGE, and black-square adherence for graded insight.

Two relaxations are forced by the data, not by preference: every english puzzle
is asymmetric and only loosely checked (0/200 are 180-symmetric or fully
checked). So `require_symmetry` is taken from each reference (False in practice)
and `min_word_len` from the shortest answer actually present -- otherwise the
target spec would be unsatisfiable for reasons the SOURCE itself violates. The
scorer's real-word / fully-checked / connected rules still gate validity, so the
model is asked to build a *stricter, well-formed* grid from the reference's word
set -- a legitimate, harder target than reproducing the loose original.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from harness.scorer import Spec, score
from pipeline.spec_generator import SpecRecord, render_spec

BLACK = "-"


@dataclass
class Puzzle:
    """One CrossWordBench puzzle reduced to what the eval needs."""

    puzzle_id: str
    difficulty: str
    size: int
    words: tuple            # exact answer set (uppercased), the runtime word_source
    n_black: int            # supplied constraint: number of black squares
    symmetric: bool         # reference black-square pattern is 180-symmetric?
    min_word_len: int       # shortest answer present (>= 2)
    density_target: float   # white / size^2 -- the black count as a density
    ref_crossings: int      # crossings in the reference grid (nuance baseline)

    def spec_record(self, spec_id_prefix="cwb") -> SpecRecord:
        return SpecRecord(
            spec_id=f"{spec_id_prefix}-{self.puzzle_id}",
            size=self.size,
            require_symmetry=self.symmetric,
            min_word_len=self.min_word_len,
            time_budget_s=float(max(5, self.size)),
            density_target=self.density_target,
            topic="general vocabulary",
            difficulty=self.difficulty or "medium",
        )

    def scorer_spec(self) -> Spec:
        # topic_words = the full supplied set, so `coverage` measures the
        # fraction of the given words the model actually placed.
        return self.spec_record().to_scorer_spec(topic_words=self.words)

    def prompt(self) -> str:
        """Program-mode SPEC (training-format renderer). The word set is passed
        to the generated program at runtime as `word_source`."""
        return render_spec(self.spec_record())

    def direct_prompt(self) -> str:
        """Direct-mode SPEC: the model emits the filled layout dict itself, so
        the exact word set is spelled out in the prompt."""
        words = ", ".join(self.words)
        sym = "with 180-degree rotational symmetry" if self.symmetric else "(symmetry not required)"
        return (
            f"Build a {self.size}x{self.size} fixed-grid crossword using EXACTLY these "
            f"{len(self.words)} words (every entry must be one of them, place as many as "
            f"you can):\n  {words}\n"
            f"Use about {self.n_black} black squares {sym}; every white run must be >= "
            f"{self.min_word_len} letters and all white cells connected.\n"
            "Output ONLY a JSON object (no prose) of the form:\n"
            '  {"rows": int, "cols": int, '
            '"cells": [{"r": int, "c": int, "letter": str, "number": int (optional)}], '
            '"across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}'
        )


def _white_cells(grid) -> set:
    return {(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != BLACK}


def _symmetric(white: set, size: int) -> bool:
    return all(((r, c) in white) == ((size - 1 - r, size - 1 - c) in white)
               for r in range(size) for c in range(size))


def _run_cells(white: set, size: int, min_len: int, horizontal: bool) -> set:
    """Cells that lie in a maximal white run of length >= min_len."""
    cells = set()
    for line in range(size):
        run = []
        for idx in range(size):
            rc = (line, idx) if horizontal else (idx, line)
            if rc in white:
                run.append(rc)
            else:
                if len(run) >= min_len:
                    cells.update(run)
                run = []
        if len(run) >= min_len:
            cells.update(run)
    return cells


def _crossings(white: set, size: int, min_len: int) -> int:
    """White cells in BOTH an across and a down run of length >= min_len.

    Matches harness.scorer's `crossings` definition, but computed from the black
    pattern alone (no letters needed) so it works on a raw reference grid.
    """
    across = _run_cells(white, size, min_len, horizontal=True)
    down = _run_cells(white, size, min_len, horizontal=False)
    return len(across & down)


def extract(row: dict) -> Puzzle:
    """Turn one raw CrossWordBench JSONL row into a Puzzle."""
    ps = json.loads(row["puzzle_state"])
    grid = ps["grid"]
    size = len(grid)
    words = tuple(sorted({str(w[0]).strip().upper() for w in ps["wordlist"]}))
    white = _white_cells(grid)
    n_black = size * size - len(white)
    min_word_len = max(2, min((len(w) for w in words), default=3))
    return Puzzle(
        puzzle_id=str(ps.get("meta_data", {}).get("id", row.get("id"))),
        difficulty=str(row.get("difficulty", "")),
        size=size,
        words=words,
        n_black=n_black,
        symmetric=_symmetric(white, size),
        min_word_len=min_word_len,
        density_target=round(len(white) / (size * size), 2),
        ref_crossings=_crossings(white, size, min_word_len),
    )


def load_puzzles(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [extract(json.loads(line)) for line in fh if line.strip()]


def score_layout(layout, puzzle: Puzzle, runtime_s=None) -> dict:
    """Score a model-produced layout under BOTH validity modes in one pass.

    The expensive step (generating + running the program) happens once upstream;
    scoring is pure, so we grade the same layout under strict (NYT) and relaxed
    (CrossWordBench-style) validity and report both `success_strict` and
    `success_relaxed`. The nuance metrics (crossings, coverage, black-square
    delta) are validity-mode-invariant, so they're taken from either pass.
    """
    spec = puzzle.scorer_spec()
    strict = score(layout, spec, word_source=puzzle.words, runtime_s=runtime_s, relaxed=False)
    relax = score(layout, spec, word_source=puzzle.words, runtime_s=runtime_s, relaxed=True)

    R = relax  # nuance fields below are identical across modes
    size = puzzle.size
    white = round(R["fill_density"] * size * size)
    black_actual = size * size - white
    return {
        "puzzle_id": puzzle.puzzle_id,
        "size": size,
        "difficulty": puzzle.difficulty,
        "status": R["status"],
        "success_strict": int(strict["valid"]),
        "success_relaxed": int(relax["valid"]),
        # --- nuance (validity-mode-invariant) ---
        "crossings": R["crossings"],
        "ref_crossings": puzzle.ref_crossings,
        "crossings_vs_ref": round(R["crossings"] / puzzle.ref_crossings, 4) if puzzle.ref_crossings else None,
        "crossing_ratio": round(R["crossings"] / white, 4) if white else 0.0,
        "coverage": R["coverage"],                 # fraction of supplied words placed
        "used_all_words": int(R["coverage"] >= 1.0),
        "black_target": puzzle.n_black,
        "black_actual": black_actual,
        "black_delta": black_actual - puzzle.n_black,
        "combined_strict": strict["combined_score"],
        "combined_relaxed": relax["combined_score"],
        "reasons_relaxed": relax["reasons"],
    }
