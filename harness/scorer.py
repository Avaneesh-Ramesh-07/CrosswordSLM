"""Deterministic scorer for fixed-grid (NYT-style) crosswords.

This is the single source of truth for the whole project: it gates dataset
examples AND scores eval outputs AND is wrapped as OpenEvolve's evaluator.

Design rule: it **trusts nothing the generator self-reports**. It reconstructs
the letter grid independently from the `across`/`down` entries, re-extracts the
maximal white runs from that grid, and re-derives every validity property from
scratch. A generator that mis-declares its entries, sneaks in an accidental
word, or reports a bogus grid cannot fool the score.

Fixed-grid / NYT rules enforced for `valid == 1`:
  - grid is exactly `size x size`
  - no crossing-letter conflicts, no out-of-bounds placement
  - every maximal white run (across AND down) has length >= min_word_len
    (this single check simultaneously enforces the minimum word length and the
     "all cells checked" rule: an unchecked cell would sit in a length-1 run)
  - the declared across/down entries equal the actual maximal runs (no
    mis-declaration, no accidental extra words)
  - every run is a real word (in word_source union the fill dictionary)
  - all white cells form one connected component
  - black squares are 180-degree rotationally symmetric (when the spec requires)

combined_score in [0, 1] is a weighted blend; `valid` carries 0.35 so an invalid
grid caps well below a valid one. `coverage` (0.30) rewards placing the target
vocabulary (the education behavior), and `fill_quality` (0.20) rewards clean,
high-score answers over crosswordese so "valid" alone can't win with junk fill.
`combined_gated = valid * raw` is reported as a robustness cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_WORD_LEN = 3
WEIGHTS = {
    "valid": 0.35,
    "coverage": 0.30,     # placing the target vocabulary IS the education behavior
    "fill_quality": 0.20,
    "density": 0.05,
    "runtime": 0.05,
    "connected": 0.05,
}
# Opt-in fitness shaping (quality_penalty=True): make "what makes a good filler"
# steer the search -- penalize filler words and any invalid connections. Off by
# default so SFT gating and the scorer unit tests are unaffected.
FILLER_PENALTY = 0.15
INVALID_PENALTY = 0.25


@dataclass
class Spec:
    """The load-bearing constraints a generated crossword must satisfy.

    Hard constraints (size, symmetry, min length) gate `valid`; density/coverage
    are graded softly; time_budget_s scores runtime when the sandbox measures it.
    """

    size: int
    topic_words: tuple = ()
    require_symmetry: bool = False  # 180-deg symmetry is an NYT convention, not a functional
    #                                 validity requirement; density + interlock are what matter
    min_word_len: int = MIN_WORD_LEN
    time_budget_s: float = 5.0
    density_target: float = 0.72
    min_vocab_fraction: float = 0.70  # >=70% of answers must be vocab n crossword-worthy


def _norm(word) -> str:
    return str(word).strip().upper()


def build_layout_from_grid(grid: dict, size: int) -> dict:
    """Build the canonical layout schema from a solved grid.

    `grid` maps (r, c) -> letter for white cells; black cells are absent.
    Produces {rows, cols, cells, across, down} with standard crossword numbering.
    Used by fixtures/seeds; the scorer re-derives runs independently regardless.
    """

    def white(r, c):
        return (r, c) in grid

    numbers = {}
    n = 0
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if not white(r, c):
                continue
            starts_a = (c == 0 or not white(r, c - 1)) and (c + 1 < size and white(r, c + 1))
            starts_d = (r == 0 or not white(r - 1, c)) and (r + 1 < size and white(r + 1, c))
            if starts_a or starts_d:
                n += 1
                numbers[(r, c)] = n
            if starts_a:
                word, cc = "", c
                while cc < size and white(r, cc):
                    word += grid[(r, cc)]
                    cc += 1
                across.append({"number": n, "row": r, "col": c, "answer": word, "len": len(word)})
            if starts_d:
                word, rr = "", r
                while rr < size and white(rr, c):
                    word += grid[(rr, c)]
                    rr += 1
                down.append({"number": n, "row": r, "col": c, "answer": word, "len": len(word)})
    cells = []
    for (r, c), ch in sorted(grid.items()):
        cell = {"r": r, "c": c, "letter": ch}
        if (r, c) in numbers:
            cell["number"] = numbers[(r, c)]
        cells.append(cell)
    return {"rows": size, "cols": size, "cells": cells, "across": across, "down": down}


def _one_component(white: set) -> bool:
    if not white:
        return False
    start = next(iter(white))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in white and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(white)


def _symmetric(white: set, size: int) -> bool:
    for r in range(size):
        for c in range(size):
            if ((r, c) in white) != ((size - 1 - r, size - 1 - c) in white):
                return False
    return True


def score(layout, spec: Spec, word_source, dictionary=None, runtime_s=None,
          scores=None, unscored_default=50, relaxed=False, vocab_set=None,
          quality_penalty=False) -> dict:
    """Score one generated crossword against `spec`. Returns a metrics dict.

    `word_source`     : iterable of allowed words (topic words + fill list).
    `dictionary`      : optional extra allowed words (broad English dictionary).
    `runtime_s`       : wall-clock seconds the generator took (from the sandbox),
                        or None to skip the runtime component.
    `scores`          : optional {WORD: 0-100} constructor-quality scores. Used
                        for `fill_quality`; if None, every answer is treated as
                        `unscored_default` (neutral, so quality neither helps nor
                        hurts). Words absent from `scores` also get the default.
    `unscored_default`: neutral score (0-100) for answers with no score.
    `relaxed`         : if True, use CrossWordBench-style validity instead of the
                        strict NYT rules: short/unchecked runs are tolerated (not
                        every cell must be checked in BOTH directions) as long as
                        every white cell still belongs to at least one real entry
                        (>= min_word_len). No-conflict, real-word, connected, and
                        dims checks still apply. Default False (strict), so the
                        OpenEvolve/training path is unchanged.
    """
    reasons: list = []
    R = {
        "status": "ok",
        "valid": 0,
        "connected": 0,
        "fill_density": 0.0,
        "fill_quality": 0.0,
        "coverage": 0.0,
        "vocab_fraction": None,
        "vocab_ok": True,
        "filler_fraction": None,
        "invalid_crossing_frac": 0.0,
        "invalid_entry_frac": 0.0,
        "runtime_ok": 1.0,
        "symmetry_ok": True,
        "accidental": 0,
        "crossings": 0,
        "n_across": 0,
        "n_down": 0,
        "n_entries": 0,
        "combined_score": 0.0,
        "combined_gated": 0.0,
        "reasons": reasons,
    }

    # --- schema ---
    if not isinstance(layout, dict) or "across" not in layout or "down" not in layout:
        R["status"] = "bad_schema"
        reasons.append("layout missing across/down")
        return R
    try:
        across = list(layout["across"])
        down = list(layout["down"])
    except TypeError:
        R["status"] = "bad_schema"
        reasons.append("across/down not iterable")
        return R

    size = spec.size
    dims_ok = layout.get("rows") == size and layout.get("cols") == size
    if not dims_ok:
        reasons.append(f"grid is not {size}x{size}")

    allowed = {_norm(w) for w in word_source}
    if dictionary:
        allowed |= {_norm(w) for w in dictionary}

    # --- reconstruct letters from the declared entries (trust nothing else) ---
    grid: dict = {}
    conflict = False
    oob = False

    def place(word, r, c, dr, dc):
        nonlocal conflict, oob
        for i, ch in enumerate(word):
            rr, cc = r + dr * i, c + dc * i
            if not (0 <= rr < size and 0 <= cc < size):
                oob = True
                return
            if (rr, cc) in grid and grid[(rr, cc)] != ch:
                conflict = True
            grid[(rr, cc)] = ch

    try:
        for e in across:
            place(_norm(e["answer"]), int(e["row"]), int(e["col"]), 0, 1)
        for e in down:
            place(_norm(e["answer"]), int(e["row"]), int(e["col"]), 1, 0)
    except (KeyError, TypeError, ValueError):
        R["status"] = "bad_schema"
        reasons.append("entry missing answer/row/col")
        return R

    if oob:
        reasons.append("entry placed out of bounds")
    if conflict:
        reasons.append("crossing letter conflict")

    white = set(grid)
    if not white:
        reasons.append("empty grid")
        return R

    # --- re-extract maximal runs (both directions), including length-1/2 runs ---
    def runs(dr, dc):
        out = []
        for (r, c) in white:
            if (r - dr, c - dc) in white:
                continue  # not the start of a run
            word, rr, cc, length = "", r, c, 0
            while (rr, cc) in white:
                word += grid[(rr, cc)]
                length += 1
                rr, cc = rr + dr, cc + dc
            out.append((r, c, word, length))
        return out

    hruns = runs(0, 1)
    vruns = runs(1, 0)
    minlen = spec.min_word_len

    bad_short = [x for x in hruns + vruns if x[3] < minlen]
    if bad_short and not relaxed:
        reasons.append(f"{len(bad_short)} run(s) shorter than {minlen} (unchecked/short)")

    # Strict compares ALL runs (so a stray length-1 run is a mismatch); relaxed
    # compares only entries >= min_word_len, treating short runs as tolerated
    # fragments on both the actual and declared sides.
    def _keep(length):
        return length >= minlen or not relaxed
    actual_a = {(r, c, w) for (r, c, w, l) in hruns if _keep(l)}
    actual_d = {(r, c, w) for (r, c, w, l) in vruns if _keep(l)}
    claimed_a = {(int(e["row"]), int(e["col"]), _norm(e["answer"]))
                 for e in across if _keep(len(_norm(e["answer"])))}
    claimed_d = {(int(e["row"]), int(e["col"]), _norm(e["answer"]))
                 for e in down if _keep(len(_norm(e["answer"])))}
    if actual_a != claimed_a:
        reasons.append("declared across entries != actual horizontal runs")
    if actual_d != claimed_d:
        reasons.append("declared down entries != actual vertical runs")

    nonword = [w for (r, c, w, l) in hruns + vruns if l >= minlen and w not in allowed]
    if nonword:
        reasons.append(f"{len(nonword)} run(s) not real words e.g. {nonword[:3]}")

    connected = _one_component(white)
    if not connected:
        reasons.append("white cells not connected")

    sym = _symmetric(white, size)
    R["symmetry_ok"] = sym
    if spec.require_symmetry and not sym:
        reasons.append("black squares not 180-degree symmetric")

    accidental = len([1 for (r, c, w, l) in hruns + vruns if l >= minlen and (r, c, w) not in (claimed_a | claimed_d)])
    R["accidental"] = accidental

    # crossings: white cells belonging to BOTH an across and a down entry
    # (each of length >= min_word_len). This is the crossword "checked-cell" /
    # word-intersection count — a graded interconnection measure that stays
    # informative even when the grid falls short of full validity.
    across_cells = {(r, c + i) for (r, c, w, l) in hruns if l >= minlen for i in range(l)}
    down_cells = {(r + i, c) for (r, c, w, l) in vruns if l >= minlen for i in range(l)}
    R["crossings"] = len(across_cells & down_cells)
    R["n_across"] = len(across)
    R["n_down"] = len(down)
    R["n_entries"] = len(across) + len(down)

    # relaxed only: every white cell must still belong to at least one real entry
    # (>= min_word_len) -- allows unchecked cells but forbids floating fragments.
    floating = white - (across_cells | down_cells)
    if relaxed and floating:
        reasons.append(f"{len(floating)} white cell(s) in no entry >= {minlen}")

    # --- metrics ---
    R["connected"] = 1 if connected else 0
    R["fill_density"] = round(len(white) / (size * size), 4)
    topic = {_norm(w) for w in spec.topic_words}
    if topic:
        placed = {w for (r, c, w, l) in hruns + vruns if l >= minlen}
        denom = max(1, min(len(topic), len(across) + len(down)))
        R["coverage"] = round(len(placed & topic) / denom, 4)
    else:
        R["coverage"] = 1.0

    # fill_quality: reward clean, high-score answers; the min term encodes the
    # crossword adage that a grid is only as good as its WORST entry.
    answers = [w for (r, c, w, l) in hruns + vruns if l >= minlen]
    if answers:
        qs = []
        for w in answers:
            s = scores.get(w) if scores else None
            qs.append((unscored_default if s is None else s) / 100.0)
        R["fill_quality"] = round(0.6 * (sum(qs) / len(qs)) + 0.4 * min(qs), 4)
    else:
        R["fill_quality"] = 0.0

    # vocab_fraction: fraction of answers that are BOTH vocabulary and crossword-worthy
    # (i.e. in the clean educational palette). This is the ">70% vocab" criterion; it
    # is distinct from `coverage`, which tracks the stricter SAT target set.
    if vocab_set is not None:
        vset = {_norm(w) for w in vocab_set}
        if answers:
            R["vocab_fraction"] = round(sum(1 for w in answers if w in vset) / len(answers), 4)
        else:
            R["vocab_fraction"] = 0.0
        R["vocab_ok"] = R["vocab_fraction"] >= spec.min_vocab_fraction
        # filler = answers that are NOT vocabulary+crossword-worthy (glue/crosswordese)
        R["filler_fraction"] = round(1.0 - R["vocab_fraction"], 4)

    # --- graded "invalid connection" metrics ------------------------------------
    # Meaningful even when the crossword is not fully valid (e.g. a model's output):
    #   (a) invalid_crossing_frac -- of all cells where an across and a down entry
    #       intersect, the fraction where the two entries disagree on the letter.
    #   (b) invalid_entry_frac    -- of all declared entries (>= min length), the
    #       fraction whose answer is not a real word (not in `allowed`).
    # is_valid == 0 whenever either is > 0 (the strict checks below subsume them).
    ga, gd = {}, {}

    def _lay(entries, dr, dc, tgt):
        for e in entries:
            try:
                r0, c0 = int(e["row"]), int(e["col"])
            except (KeyError, ValueError, TypeError):
                continue
            for i, ch in enumerate(_norm(e["answer"])):
                rr, cc = r0 + dr * i, c0 + dc * i
                if 0 <= rr < size and 0 <= cc < size:
                    tgt[(rr, cc)] = ch

    _lay(across, 0, 1, ga)
    _lay(down, 1, 0, gd)
    cross = set(ga) & set(gd)
    if cross:
        R["invalid_crossing_frac"] = round(sum(1 for c in cross if ga[c] != gd[c]) / len(cross), 4)
    declared = [e for e in across + down if len(_norm(e["answer"])) >= minlen]
    if declared:
        bad = sum(1 for e in declared if _norm(e["answer"]) not in allowed)
        R["invalid_entry_frac"] = round(bad / len(declared), 4)

    if runtime_s is not None:
        b = spec.time_budget_s
        if runtime_s <= 0.4 * b:
            R["runtime_ok"] = 1.0
        elif runtime_s >= b:
            R["runtime_ok"] = 0.0
        else:
            R["runtime_ok"] = round(max(0.0, 1 - (runtime_s - 0.4 * b) / (0.6 * b)), 4)

    valid = (
        not conflict
        and not oob
        and (relaxed or not bad_short)
        and (not relaxed or not floating)
        and actual_a == claimed_a
        and actual_d == claimed_d
        and not nonword
        and connected
        and (sym or not spec.require_symmetry)
        and dims_ok
    )
    R["valid"] = 1 if valid else 0

    penalty = min(0.15, 0.05 * accidental)
    raw = (
        WEIGHTS["valid"] * R["valid"]
        + WEIGHTS["fill_quality"] * R["fill_quality"]
        + WEIGHTS["coverage"] * R["coverage"]
        + WEIGHTS["density"] * R["fill_density"]
        + WEIGHTS["runtime"] * R["runtime_ok"]
        + WEIGHTS["connected"] * R["connected"]
    )
    if quality_penalty:
        raw -= FILLER_PENALTY * (R["filler_fraction"] or 0.0)
        raw -= INVALID_PENALTY * (R["invalid_crossing_frac"] + R["invalid_entry_frac"])
    raw = max(0.0, min(1.0, raw - penalty))
    R["combined_score"] = round(raw, 4)
    R["combined_gated"] = round(R["valid"] * raw, 4)
    return R
