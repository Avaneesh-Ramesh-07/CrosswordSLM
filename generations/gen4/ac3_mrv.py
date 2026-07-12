"""gen4 fusion: AC-3 / MAC arc-consistency + MRV + pattern-index, palette-scaled.

Parents: csp_ac3, 602461b0.
Technique: bounded AC-3 preprocessing prunes each slot's (capped) word domain to
arc-consistent values (dropping words with no compatible neighbour at a shared
cell); search maintains arc consistency (MAC) after every placement with a
trail-based O(pruned) undo, MRV slot ordering, and pattern-index frequency value
ordering. Domains are capped to a few thousand and revision stops at the deadline
so it stays fast on the big palette.

PALETTE-SCALING FIX: the pattern index is built ONCE and reused across structure
attempts; the real per-size deadline is used; sizes 9/11 use pre-verified fillable
templates (random symmetric grids at those sizes almost never fill from the full
palette).

Self-contained, stdlib only; never hardcodes words -- every answer comes from
word_source (list, or {"theme","fill"} dict). Returns the standard layout schema.
"""

import random
import time
from collections import deque

# Pre-verified-fillable black-square templates (11x11 robust-first, 9x9). Random
# symmetric grids at 9x9/11x11 rarely fill from the full clean palette, so these
# grids are SELECTED (harvested from verified-good size-9/11 generators + a
# fillability search against the clean palette); size 7 uses random construction.
_TEMPLATES_11 = [
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 5], [3, 6], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [7, 4], [7, 5], [8, 6], [9, 6], [10, 6], [10, 10]],
    [[0, 4], [0, 10], [1, 4], [2, 4], [3, 6], [4, 3], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [6, 7], [7, 4], [8, 6], [9, 6], [10, 0], [10, 6]],
    [[0, 6], [1, 6], [2, 6], [3, 0], [3, 1], [3, 5], [3, 9], [3, 10], [4, 4], [5, 3], [5, 7], [6, 6], [7, 0], [7, 1], [7, 5], [7, 9], [7, 10], [8, 4], [9, 4], [10, 4]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [3, 0], [3, 5], [3, 9], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 1], [7, 5], [7, 10], [9, 3], [9, 7], [10, 3], [10, 7]],
    [[0, 4], [0, 5], [0, 10], [1, 4], [2, 4], [3, 3], [3, 7], [4, 6], [5, 0], [5, 1], [5, 9], [5, 10], [6, 4], [7, 3], [7, 7], [8, 6], [9, 6], [10, 0], [10, 5], [10, 6]],
    [[0, 3], [0, 4], [1, 4], [2, 4], [3, 0], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [7, 10], [8, 6], [9, 6], [10, 6], [10, 7]],
    [[0, 0], [0, 4], [0, 5], [1, 4], [3, 7], [4, 0], [4, 1], [4, 2], [4, 6], [5, 0], [5, 10], [6, 4], [6, 8], [6, 9], [6, 10], [7, 3], [9, 6], [10, 5], [10, 6], [10, 10]],
    [[0, 3], [0, 7], [3, 0], [3, 5], [3, 6], [4, 0], [4, 1], [4, 2], [4, 6], [5, 3], [5, 7], [6, 4], [6, 8], [6, 9], [6, 10], [7, 4], [7, 5], [7, 10], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [3, 0], [3, 1], [3, 2], [3, 6], [3, 10], [4, 3], [5, 3], [5, 7], [6, 7], [7, 0], [7, 4], [7, 8], [7, 9], [7, 10], [9, 3], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [2, 3], [3, 0], [3, 1], [3, 5], [3, 10], [4, 4], [6, 6], [7, 0], [7, 5], [7, 9], [7, 10], [8, 7], [9, 3], [9, 7], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 3], [3, 5], [3, 6], [4, 0], [4, 6], [4, 7], [5, 0], [5, 1], [5, 9], [5, 10], [6, 3], [6, 4], [6, 10], [7, 4], [7, 5], [9, 7], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [2, 7], [3, 4], [3, 5], [3, 6], [4, 3], [4, 10], [5, 0], [5, 10], [6, 0], [6, 7], [7, 4], [7, 5], [7, 6], [8, 3], [9, 3], [10, 3], [10, 7]],
    [[0, 4], [0, 5], [1, 5], [3, 0], [3, 1], [3, 6], [3, 10], [4, 6], [4, 7], [5, 3], [5, 7], [6, 3], [6, 4], [7, 0], [7, 4], [7, 9], [7, 10], [9, 5], [10, 5], [10, 6]],
    [[0, 3], [0, 7], [1, 3], [3, 5], [4, 6], [4, 7], [5, 0], [5, 1], [5, 2], [5, 3], [5, 7], [5, 8], [5, 9], [5, 10], [6, 3], [6, 4], [7, 5], [9, 7], [10, 3], [10, 7]],
    [[0, 0], [0, 5], [1, 5], [3, 4], [3, 8], [3, 9], [3, 10], [4, 4], [4, 10], [5, 3], [5, 7], [6, 0], [6, 6], [7, 0], [7, 1], [7, 2], [7, 6], [9, 5], [10, 5], [10, 10]],
    [[0, 6], [0, 7], [1, 6], [3, 4], [3, 5], [4, 0], [4, 4], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 6], [6, 10], [7, 5], [7, 6], [9, 4], [10, 3], [10, 4]],
    [[0, 3], [0, 7], [1, 7], [2, 7], [3, 5], [3, 6], [4, 0], [4, 1], [4, 2], [4, 6], [6, 4], [6, 8], [6, 9], [6, 10], [7, 4], [7, 5], [8, 3], [9, 3], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [3, 0], [3, 5], [3, 6], [4, 0], [4, 1], [4, 6], [5, 3], [5, 7], [6, 4], [6, 9], [6, 10], [7, 4], [7, 5], [7, 10], [9, 3], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [2, 7], [3, 5], [3, 6], [4, 3], [4, 10], [5, 0], [5, 1], [5, 9], [5, 10], [6, 0], [6, 7], [7, 4], [7, 5], [8, 3], [9, 3], [10, 3], [10, 7]],
    [[0, 0], [0, 5], [1, 0], [1, 5], [2, 5], [3, 4], [3, 9], [3, 10], [4, 7], [5, 3], [5, 7], [6, 3], [7, 0], [7, 1], [7, 6], [8, 5], [9, 5], [9, 10], [10, 5], [10, 10]],
    [[0, 0], [0, 1], [0, 7], [1, 7], [3, 5], [3, 10], [4, 4], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 6], [7, 0], [7, 5], [9, 3], [10, 3], [10, 9], [10, 10]],
    [[0, 5], [0, 9], [0, 10], [1, 5], [3, 3], [3, 7], [4, 0], [4, 6], [5, 0], [5, 1], [5, 9], [5, 10], [6, 4], [6, 10], [7, 3], [7, 7], [9, 5], [10, 0], [10, 1], [10, 5]],
    [[0, 3], [0, 7], [3, 5], [3, 10], [4, 3], [4, 4], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [6, 6], [6, 7], [7, 0], [7, 5], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [2, 7], [3, 0], [3, 1], [3, 5], [4, 0], [4, 4], [5, 0], [5, 10], [6, 6], [6, 10], [7, 5], [7, 9], [7, 10], [8, 3], [9, 3], [10, 3], [10, 7]],
    [[0, 3], [0, 4], [1, 3], [2, 3], [3, 0], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [7, 10], [8, 7], [9, 7], [10, 6], [10, 7]],
    [[0, 4], [1, 4], [2, 4], [3, 0], [3, 1], [3, 5], [3, 6], [3, 10], [4, 6], [4, 7], [6, 3], [6, 4], [7, 0], [7, 4], [7, 5], [7, 9], [7, 10], [8, 6], [9, 6], [10, 6]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [3, 4], [3, 5], [3, 6], [4, 0], [4, 1], [5, 3], [5, 7], [6, 9], [6, 10], [7, 4], [7, 5], [7, 6], [9, 3], [9, 7], [10, 3], [10, 7]],
    [[0, 6], [0, 7], [1, 6], [3, 0], [3, 1], [3, 5], [3, 9], [3, 10], [4, 3], [4, 4], [6, 6], [6, 7], [7, 0], [7, 1], [7, 5], [7, 9], [7, 10], [9, 4], [10, 3], [10, 4]],
    [[0, 3], [0, 7], [1, 3], [3, 0], [3, 1], [3, 5], [3, 6], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 4], [7, 5], [7, 9], [7, 10], [9, 7], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [3, 4], [3, 5], [3, 6], [4, 3], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 7], [7, 4], [7, 5], [7, 6], [9, 3], [10, 3], [10, 7]],
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 5], [3, 6], [3, 10], [4, 3], [4, 10], [5, 3], [5, 7], [6, 0], [6, 7], [7, 0], [7, 4], [7, 5], [8, 6], [9, 6], [10, 6], [10, 10]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [3, 0], [3, 5], [3, 6], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 4], [7, 5], [7, 10], [9, 3], [9, 7], [10, 3], [10, 7]],
    [[0, 0], [0, 5], [1, 5], [2, 5], [3, 6], [4, 3], [4, 10], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 0], [6, 7], [7, 4], [8, 5], [9, 5], [10, 5], [10, 10]],
    [[0, 0], [0, 4], [1, 0], [1, 4], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [5, 3], [5, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [9, 6], [9, 10], [10, 6], [10, 10]],
    [[0, 3], [0, 7], [1, 7], [3, 5], [3, 10], [4, 3], [4, 4], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 6], [6, 7], [7, 0], [7, 5], [9, 3], [10, 3], [10, 7]],
    [[0, 3], [0, 4], [0, 5], [1, 5], [2, 5], [3, 7], [4, 4], [4, 9], [4, 10], [5, 0], [5, 10], [6, 0], [6, 1], [6, 6], [7, 3], [8, 5], [9, 5], [10, 5], [10, 6], [10, 7]],
    [[0, 6], [1, 6], [2, 6], [3, 0], [3, 1], [3, 5], [3, 9], [3, 10], [4, 3], [5, 3], [5, 7], [6, 7], [7, 0], [7, 1], [7, 5], [7, 9], [7, 10], [8, 4], [9, 4], [10, 4]],
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [5, 3], [5, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [8, 6], [9, 6], [10, 6], [10, 10]],
    [[0, 3], [0, 4], [1, 3], [3, 5], [3, 6], [3, 10], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [7, 0], [7, 4], [7, 5], [9, 7], [10, 6], [10, 7]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [3, 0], [3, 1], [3, 5], [4, 0], [4, 6], [5, 3], [5, 7], [6, 4], [6, 10], [7, 5], [7, 9], [7, 10], [9, 3], [9, 7], [10, 3], [10, 7]],
    [[0, 4], [0, 5], [0, 6], [1, 4], [3, 3], [3, 7], [4, 0], [4, 1], [4, 6], [5, 0], [5, 10], [6, 4], [6, 9], [6, 10], [7, 3], [7, 7], [9, 6], [10, 4], [10, 5], [10, 6]],
    [[0, 4], [0, 5], [1, 5], [2, 5], [3, 0], [3, 6], [3, 10], [4, 3], [4, 7], [5, 3], [5, 7], [6, 3], [6, 7], [7, 0], [7, 4], [7, 10], [8, 5], [9, 5], [10, 5], [10, 6]],
    [[0, 3], [0, 7], [1, 7], [3, 0], [3, 1], [3, 5], [3, 6], [3, 10], [4, 3], [5, 3], [5, 7], [6, 7], [7, 0], [7, 4], [7, 5], [7, 9], [7, 10], [9, 3], [10, 3], [10, 7]],
    [[0, 6], [1, 6], [2, 6], [3, 0], [3, 4], [3, 5], [4, 0], [4, 1], [4, 2], [5, 3], [5, 7], [6, 8], [6, 9], [6, 10], [7, 5], [7, 6], [7, 10], [8, 4], [9, 4], [10, 4]],
    [[0, 3], [0, 7], [1, 3], [2, 3], [3, 0], [3, 4], [3, 5], [3, 9], [3, 10], [4, 4], [6, 6], [7, 0], [7, 1], [7, 5], [7, 6], [7, 10], [8, 7], [9, 7], [10, 3], [10, 7]],
    [[0, 3], [0, 7], [1, 7], [3, 4], [3, 5], [4, 0], [4, 4], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 6], [6, 10], [7, 5], [7, 6], [9, 3], [10, 3], [10, 7]],
]

_TEMPLATES_9 = [
    [[0, 0], [0, 4], [0, 8], [1, 4], [2, 4], [3, 3], [4, 0], [4, 8], [5, 5], [6, 4], [7, 4], [8, 0], [8, 4], [8, 8]],
    [[0, 4], [3, 3], [3, 7], [3, 8], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 0], [5, 1], [5, 5], [8, 4]],
    [[0, 5], [1, 5], [3, 0], [3, 1], [3, 2], [3, 3], [4, 0], [4, 8], [5, 5], [5, 6], [5, 7], [5, 8], [7, 3], [8, 3]],
    [[0, 4], [0, 8], [1, 4], [2, 4], [3, 0], [3, 1], [3, 5], [5, 3], [5, 7], [5, 8], [6, 4], [7, 4], [8, 0], [8, 4]],
    [[0, 3], [0, 4], [1, 3], [1, 4], [2, 4], [3, 0], [3, 5], [5, 3], [5, 8], [6, 4], [7, 4], [7, 5], [8, 4], [8, 5]],
    [[0, 3], [0, 4], [0, 8], [1, 4], [3, 5], [4, 0], [4, 1], [4, 7], [4, 8], [5, 3], [7, 4], [8, 0], [8, 4], [8, 5]],
    [[0, 4], [1, 4], [3, 3], [3, 8], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 0], [5, 5], [7, 4], [8, 4]],
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 3], [3, 8], [4, 0], [4, 8], [5, 0], [5, 5], [6, 4], [7, 4], [8, 4], [8, 8]],
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 3], [3, 7], [3, 8], [5, 0], [5, 1], [5, 5], [6, 4], [7, 4], [8, 4], [8, 8]],
    [[0, 3], [0, 4], [0, 8], [1, 3], [3, 0], [3, 1], [3, 5], [5, 3], [5, 7], [5, 8], [7, 5], [8, 0], [8, 4], [8, 5]],
    [[0, 0], [0, 4], [0, 8], [3, 3], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 5], [8, 0], [8, 4], [8, 8]],
    [[0, 3], [0, 4], [0, 8], [1, 4], [2, 4], [3, 5], [4, 0], [4, 8], [5, 3], [6, 4], [7, 4], [8, 0], [8, 4], [8, 5]],
    [[0, 0], [0, 4], [0, 8], [3, 5], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 3], [8, 0], [8, 4], [8, 8]],
    [[0, 3], [0, 8], [3, 0], [3, 5], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 3], [5, 8], [8, 0], [8, 5]],
    [[0, 0], [0, 4], [1, 4], [3, 3], [3, 8], [4, 0], [4, 1], [4, 7], [4, 8], [5, 0], [5, 5], [7, 4], [8, 4], [8, 8]],
    [[0, 0], [0, 4], [1, 4], [3, 3], [3, 7], [3, 8], [4, 0], [4, 8], [5, 0], [5, 1], [5, 5], [7, 4], [8, 4], [8, 8]],
    [[0, 4], [1, 4], [2, 4], [3, 0], [3, 5], [4, 0], [4, 1], [4, 7], [4, 8], [5, 3], [5, 8], [6, 4], [7, 4], [8, 4]],
    [[0, 4], [1, 4], [2, 4], [3, 3], [3, 8], [4, 0], [4, 1], [4, 7], [4, 8], [5, 0], [5, 5], [6, 4], [7, 4], [8, 4]],
]


def _split_source(word_source):
    """Accept a plain list OR a {"theme","fill"} dict. -> (theme, fill) upper lists."""
    if isinstance(word_source, dict):
        theme = [str(w).upper() for w in word_source.get("theme", [])]
        fill = [str(w).upper() for w in word_source.get("fill", [])]
        return theme, fill
    return [], [str(w).upper() for w in word_source]


def _index_by_length(word_source):
    idx = {}
    for w in word_source:
        w = str(w).upper()
        if w.isalpha():
            idx.setdefault(len(w), []).append(w)
    return idx


def _runs(white, size):
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(size):
            for c in range(size):
                if (r, c) not in white:
                    continue
                if (r - dr, c - dc) in white:
                    continue
                cells, rr, cc = [], r, c
                while (rr, cc) in white:
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                out.append((cells, len(cells)))
    return out


def _connected(white):
    if not white:
        return False
    start = next(iter(white))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in white and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(white)


def _structure_ok(white, size, min_len=3):
    if not white:
        return False
    if any(length < min_len for _, length in _runs(white, size)):
        return False
    return _connected(white)


def _make_structure(size, rng, min_len=3):
    """Verbatim reference_v1 structure builder (used for size 7)."""
    full = {(r, c) for r in range(size) for c in range(size)}
    if size <= 5:
        return full
    cells = list(full)
    for _ in range(60):
        rng.shuffle(cells)
        blacks = set()
        target = (size * size) // 6
        for (r, c) in cells:
            if len(blacks) >= target:
                break
            partner = (size - 1 - r, size - 1 - c)
            if (r, c) == partner or (r, c) in blacks or partner in blacks:
                continue
            trial_white = full - (blacks | {(r, c), partner})
            if _structure_ok(trial_white, size, min_len):
                blacks |= {(r, c), partner}
        white = full - blacks
        if _structure_ok(white, size, min_len):
            return white
    return full


def _slots_and_crossings(white, size):
    slots = []
    for cells, length in _runs(white, size):
        slots.append({"cells": cells, "len": length})
    cell_to_slots = {}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            cell_to_slots.setdefault(cell, []).append(i)
    return slots, cell_to_slots


def _build_pattern_index(idx_by_len):
    """(length, position, letter) -> set(words). Built ONCE per call and reused
    across every structure attempt (the palette-scaling fix)."""
    pat = {}
    for length, words in idx_by_len.items():
        for w in words:
            for pos, ch in enumerate(w):
                pat.setdefault((length, pos, ch), set()).add(w)
    return pat


def _crossings(slots):
    """For each slot: list of (my_pos, other_slot, other_pos) at each shared cell,
    plus a parallel list of slot lengths. In an all-checked grid every white cell
    is in exactly one across + one down slot, so each shared cell yields one arc."""
    cellpos = {}
    for si, s in enumerate(slots):
        for pos, cell in enumerate(s["cells"]):
            cellpos.setdefault(cell, []).append((si, pos))
    cross = [[] for _ in range(len(slots))]
    for lst in cellpos.values():
        if len(lst) == 2:
            (a, pa), (b, pb) = lst
            cross[a].append((pa, b, pb))
            cross[b].append((pb, a, pa))
    return cross, [s["len"] for s in slots]


def _build_layout(white, size, slots, assignment):
    grid = {}
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            grid[(r, c)] = word[pos]

    def is_white(r, c):
        return (r, c) in grid

    numbers, n = {}, 0
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if not is_white(r, c):
                continue
            sa = (c == 0 or not is_white(r, c - 1)) and (c + 1 < size and is_white(r, c + 1))
            sd = (r == 0 or not is_white(r - 1, c)) and (r + 1 < size and is_white(r + 1, c))
            if sa or sd:
                n += 1
                numbers[(r, c)] = n
            if sa:
                w, cc = "", c
                while cc < size and is_white(r, cc):
                    w += grid[(r, cc)]
                    cc += 1
                across.append({"number": n, "row": r, "col": c, "answer": w, "len": len(w)})
            if sd:
                w, rr = "", r
                while rr < size and is_white(rr, c):
                    w += grid[(rr, c)]
                    rr += 1
                down.append({"number": n, "row": r, "col": c, "answer": w, "len": len(w)})
    cells = []
    for (r, c), ch in sorted(grid.items()):
        cell = {"r": r, "c": c, "letter": ch}
        if (r, c) in numbers:
            cell["number"] = numbers[(r, c)]
        cells.append(cell)
    return {"rows": size, "cols": size, "cells": cells, "across": across, "down": down}


_CAP = 2000  # domain bound: cap each slot's word list before propagation


def _ac3(dom, cross, deadline):
    """Full arc-consistency pass; prunes words with no support at a shared cell."""
    q = deque((si, mp, nsi, npos) for si in range(len(dom)) for (mp, nsi, npos) in cross[si])
    inq = set(q)
    while q:
        if time.perf_counter() > deadline:
            return True  # bounded: stop revising at the deadline, keep what we have
        arc = q.popleft()
        inq.discard(arc)
        si, mp, nsi, npos = arc
        supp = {w[npos] for w in dom[nsi]}
        nd = {w for w in dom[si] if w[mp] in supp}
        if len(nd) != len(dom[si]):
            if not nd:
                return False
            dom[si] = nd
            for (mp2, nsi2, npos2) in cross[si]:
                if nsi2 != nsi:
                    a2 = (nsi2, npos2, si, mp2)
                    if a2 not in inq:
                        q.append(a2)
                        inq.add(a2)
    return True


def _mac(dom, cross, deadline, start, trail):
    """Maintain arc consistency after fixing `start`; record first-change old
    domains in `trail` for O(pruned) undo (no full-domain snapshots)."""
    q = deque((nsi, npos, start, mp) for (mp, nsi, npos) in cross[start])
    inq = set(q)
    while q:
        if time.perf_counter() > deadline:
            return True
        arc = q.popleft()
        inq.discard(arc)
        si, mp, nsi, npos = arc
        supp = {w[npos] for w in dom[nsi]}
        nd = {w for w in dom[si] if w[mp] in supp}
        if len(nd) != len(dom[si]):
            if si not in trail:
                trail[si] = dom[si]
            dom[si] = nd
            if not nd:
                return False
            for (mp2, nsi2, npos2) in cross[si]:
                if nsi2 != nsi:
                    a2 = (nsi2, npos2, si, mp2)
                    if a2 not in inq:
                        q.append(a2)
                        inq.add(a2)
    return True


def _solve(slots, cross, slen, by_len, pat, LF, rng, deadline, theme_set):
    """AC-3 preprocessing + MAC (maintained arc consistency) + MRV over bounded
    set-domains, pattern-index frequency value order."""
    n = len(slots)
    dom = []
    for s in slots:
        p = list(by_len.get(s["len"], []))
        rng.shuffle(p)
        dom.append(set(p[:_CAP]))
    if not _ac3(dom, cross, deadline):
        return None
    assign, used = {}, set()

    def bt():
        if time.perf_counter() > deadline:
            return None
        if len(assign) == n:
            return True
        best, bs = -1, 1 << 30
        for si in range(n):
            if si in assign:
                continue
            l = len(dom[si])
            if l < bs:
                bs, best = l, si
        if bs == 0:
            return False
        si = best
        crs = cross[si]
        cands = [w for w in dom[si] if w not in used]
        cands.sort(key=lambda w: sum(LF.get((slen[nsi], npos, w[mp]), 0)
                   for (mp, nsi, npos) in crs if nsi not in assign), reverse=True)
        for w in cands:
            if time.perf_counter() > deadline:
                return None
            trail = {si: dom[si]}
            dom[si] = {w}
            ok = _mac(dom, cross, deadline, si, trail)
            if ok:
                assign[si] = w
                used.add(w)
                r = bt()
                if r is True:
                    return True
                del assign[si]
                used.discard(w)
                if r is None:
                    for x, old in trail.items():
                        dom[x] = old
                    return None
            for x, old in trail.items():
                dom[x] = old
        return False

    return dict(assign) if bt() is True else None


def _structures(size, rng):
    """Yield candidate white-cell sets. Size 7 -> random symmetric construction;
    sizes 9/11 -> pre-verified fillable templates (cycled for extra randomization)."""
    full = {(r, c) for r in range(size) for c in range(size)}
    if size == 11:
        tpls = _TEMPLATES_11
    elif size == 9:
        tpls = _TEMPLATES_9
    else:
        tpls = None
    if tpls is None:
        while True:
            yield _make_structure(size, rng)
    else:
        while True:
            for black in tpls:
                yield full - {(r, c) for (r, c) in black}


def generate_crossword(topic, word_source, size):
    budgets = {5: 2.0, 7: 3.0, 9: 5.0, 11: 12.0, 13: 20.0, 15: 30.0}
    per_attempt = {5: 0.6, 7: 0.7, 9: 1.1, 11: 1.4}.get(size, 1.2)
    deadline = time.perf_counter() + budgets.get(size, 5.0) - 0.4
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    by_len = _index_by_length(theme + fill)
    empty = {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
    if not by_len:
        return empty
    # PALETTE-SCALING FIX: build the (length,pos,letter) pattern index ONCE and
    # reuse it across every structure attempt; LF is the per-bucket size used for
    # value ordering. (The parents rebuilt this inside the structure loop.)
    pat = _build_pattern_index(by_len)
    LF = {k: len(v) for k, v in pat.items()}
    for white in _structures(size, rng):
        if time.perf_counter() > deadline:
            break
        slots, _ = _slots_and_crossings(white, size)
        cross, slen = _crossings(slots)
        sub = min(deadline, time.perf_counter() + per_attempt)
        assignment = _solve(slots, cross, slen, by_len, pat, LF, rng, sub, theme_set)
        if assignment and len(assignment) == len(slots):
            return _build_layout(white, size, slots, assignment)
    return empty


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from pipeline.eval_selfmodel import english_palette
    pal = english_palette(11)
    for sz in (7, 9, 11):
        t0 = time.perf_counter()
        lay = generate_crossword("vocabulary", pal["ws"], sz)
        print(f"size {sz}: {len(lay['across'])}A {len(lay['down'])}D, "
              f"{len(lay['cells'])} white cells  ({time.perf_counter()-t0:.1f}s)")
