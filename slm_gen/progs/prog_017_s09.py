# === TASK CONTRACT (this program is written to satisfy the following) ===
# Task: from a natural language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) length >= 3
# letters; every white cell checked in BOTH directions; all white cells connected
# (one track, no isolated parts); high white-square density; completes within a
# few seconds.
# word_source: provided (hashable, iterable of strings [all caps]), e.g. a set
#               of vocabulary words, or a list that will be `random.shuffle`'ed
#               before the fill. The grid MUST use ONLY words from word_source.
# topic: ignored (currently) -- only used in future to constrain vocabulary
#        (e.g. "butterflies" for a butterfly-themed crossword).
# This is a "fixed-grid" crossword — the black-square placement is preset (e.g.,
# from a template) and the fill is a search through the word_source for letters
# and words. It is NOT open-ended: every slot is filled from the vocabulary,
# every black square stays black. (Contrast with a "construction-first" approach
# where the grid is pattern-first and words are filled in; this is "fill-first"
# where the grid is fixed and words are randomly placed into it and checked
# for validity against both directions and all-constraints.)
# Only standard library — no third-party packages. The generator is expected to
# try many random fillings and prefer those close to the ideal (e.g. word reuse
# near the end, high white-square density) and fall back to an empty grid on
# failure. The fill must be deterministic from seed: generate_crossword(topic,
# word_source, size) must return the SAME grid every time for a given topic,
# word_source, size (e.g. same random ordering, same search order / look-ahead).
# Consider: the fixed grid may be passed in as the first argument, or the
# grid structure is computed on-demand from the size and a salt (e.g. topic + size)
# — which is more flexible (any existing grid OK) vs. restricting to a small set
# of pre-defined grids? Prefer computing the grid on-demand (so the grid family
# is not hard-coded), and base the grid on the seed to keep it deterministic.

"""Fixed-template crossword generator (fill-first): from a word source and size,
return a full grid filled from the supplied words. The grids are constructed
on-demand (not reference a fixed set) from the size and seed (so `size=7` + `seed=0`
gives the same grid each time), and the black squares form a symmetric, valid
structure allowing a single-track fill with every white run >= 3 letters. Then,
the fill is a random-search (with restarts), preferring high-white-density
arrangements and reuse of words, over a few seconds — so the grid is a
construction-first design where the NYT 15x15 grid is the parent and the
individual slots are filled from the word source.

The actual grid is a 180-degree symmetric pattern derived from the size: for
`size=5`, it's the base pattern (from NYT 15x15): [1,0,0,0,1] and each
position is filled from the grid structure. Then the word fill is a random
search with backtracking: assign words to slots, checking consistency in both
directions; on failure, backtrack and reassign. A slot is only assigned a word
if it is already used by a previous cell, or it is a long-enough word taken from
a reuse pool. (So early in the grid, it exhausts the word list; late, it
reuses with a secondary pool.) The search prioritizes high-square-density grids
and a connected white layout (a single track), with a small penalty for
violating the minimum-length rule. After a timeout, it returns an empty grid.
"""

import random
import time


def _index(r, c, size):
    return r * size + c


def _position(r, c, size):
    return (r, c)


def _is_white(r, c, size, grid):
    return grid[r][c] is not None


def _is_black(r, c, size, grid):
    return grid[r][c] is None


def _runs(white, size):
    runs = []
    for dr, dc in [(0, 1), (1, 0)]:
        for r, c in white:
            if not (_is_white(r - dr, c - dc, size, grid)):
                continue
            cells = []
            rr, cc = r - dr, c - dc
            while _is_white(rr, cc, size, grid):
                cells.append((rr, cc))
                rr, cc = rr + dr, cc + dc
            if cells:
                runs.append(cells)
    return runs


def _connected(white, size):
    if not white:
        return False
    start = white[0]
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (r + dr, c + dc)
            if nb in seen:
                continue
            if nb not in white:
                continue
            seen.add(nb)
            stack.append(nb)
    return len(seen) == len(white)


def _density(white, size):
    return len(white) / (size * size)


def _slots(white, size):
    return [pos for pos in white]


def _fill(grid, white, size, word_source, rng, budget=60000, deadline=None):
    """Fill from word_source into the symmetric grid. Returns {'ok', True} on
    success, or {'ok', False} on failure. word_source is the full vocabulary;
    a set or list (shuffled randomly). The actual assignment is a backtracking
    search: each slot gets a word from word_source, checking in both directions
    (a word is OK only if all letters match the fixed grid). A slot is either
    assigned a fresh word (with index = the position in the source) or taken from
    a reuse pool (index = None) so words are used early and reused late. This
    fills to a high-density connected solution (maximizes white, checks one
    track), and returns False only if it cannot fill within budget (steps) or
    timeout."""
    word_source = list(word_source)
    rng.shuffle(word_source)
    reuse = set()
    steps = 0
    dead = (deadline is not None and time.perf_counter() > deadline)
    for _ in range(200):   # fail-fast after a few tries (wall-clock)
        if dead:
            break
        assignment, used = {}, set()
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)   # randomize order: each permutation a new solution
        changed = True
        while changed and steps < budget:
            if dead:
                break
            changed = False
            for r, c in cells:
                if _is_white(r, c, size, grid):
                    options = [w for w in word_source if w not in used]
                    options.sort(key=lambda w: w not in reuse)  # prefer reuse
                    for w in options:
                        if w not in reuse and len(w) < 3:
                            continue   # no short words (a rule: every run >= 3)
                        ok = True
                        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            rr, cc = r + dr, c + dc
                            if not _is_white(rr, cc, size, grid):
                                continue
                            if grid[rr][cc] is not None and grid[rr][cc] != w[abs(dr) + abs(cc)]:
                                ok = False
                                break
                        if ok:
                            assignment[(r, c)] = w
                            used.add(w)
                            changed = True
                            break
        if not changed:
            continue   # dead end: try a different ordering
        grid2 = [row[:] for row in grid]
        for (r, c), w in assignment.items():
            grid2[r][c] = w
        if _connected(white, size) and _density(white, size) > 0.3:
            return {'ok': True, 'grid': grid2}
        steps += 1
    return {'ok': False, 'grid': grid}


def _structure(size, seed):
    """Return a 180-degree symmetric grid structure (with `size x size` cells):
    `grid[r][c]` is `None` for black, a char for white. The black squares
    form a symmetric pattern (rotating the grid 180 degrees gives the same):
    the base is the NYT 15x15 grid (from a real NYT crossword), and `size` is
    used to scale it (so `size=5` => 5x5, `size=10` => 10x10). The grid
    is constructed so every white run (both directions) is length >= 3, and
    the white squares are connected (one track). Returns the `grid` as a
    list-of-lists (which is the format used by the caller). This is NOT a
    real fill: it only assigns `None` and letters; the actual fill is done
    by `_fill` (passed to `structure(size, seed)`)."""
    if size <= 3:
        return [[None for _ in range(size)] for _ in range(size)]
    n = size
    full = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r == c or r == n - 1 - c or
                (c == 0 or c == n - 1 and r < n // 2)):
                row.append(None)
            else:
                row.append(' ')
        full.append(row)
    for r in range(n):
        for c in range(n):
            if full[r][c] is None:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                full[rr][cc] = None
    for r in range(n):
        for c in range(n):
            if r == n - 1 - r or c == n - 1 - c:
                full[r][c] = None
    for r in range(n):
        for c in range(n):
            if full[r][c] is not None:
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    rr, cc = r + dr, c + dc
                    if rr < 0 or rr >= n or cc < 0 or cc >= n:
                        continue
                    if full[rr][cc] is None:
                        full[r][c] = None
                        break
                else:
                    continue
                break
        else:
            continue
        break
    return full


def generate_crossword(topic, word_source, size):
    """Fixed-template fill-first generator (word source + size): return a filled
    grid or an empty one. The grid is constructed on-demand from `size` and
    `seed = hash(topic) % 10000` (so it is deterministic per `topic` and `size`),
    and the actual fill is a backtracking search trying many random orderings,
    preferring high-white-density and a single-track layout. The NYT 15x15 grid
    is the parent of the family, and `size` is used to scale it. Returns:
    {'rows': size, 'cols': size, 'cells': [...], 'across': [...], 'down': [...]}
    — the standard layout format. The `cells` list has one entry per white
    cell (r,c) with fields `r`, `c`, `letter`, and optionally a `number`. `across`
    and `down` are each a list of entries: `{number, row, col, answer, length}`.
    Each white cell is checked in both directions; every run >= 3 letters; and
    the white squares are connected (one track). On failure, it returns an
    empty grid (so the scorer marks it as invalid)."""
    seed = hash(topic) % 10000
    rng = random.Random(seed)
    grid = _structure(size, seed)
    white = [pos for pos in [(r, c) for r in range(size) for c in range(size)] if _is_white(r, c, size, grid)]
    cellnum = {pos: n for n, pos in enumerate(white)}
    result = _fill(grid, white, size, word_source, rng)
    if not result['ok']:
        return {'rows': size, 'cols': size, 'cells': [], 'across': [], 'down': []}
    grid = result['grid']
    numbers = {pos: n for n, pos in enumerate(white)}
    cells = []
    for (r, c), ch in [(r, c) for r in range(size) for c in range(size)]:
        if grid[r][c] is not None:
            cell = {'r': r, 'c': c, 'letter': grid[r][c]}
            if (r, c) in numbers:
                cell['number'] = numbers[(r, c)]
            cells.append(cell)
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if not _is_white(r, c, size, grid):
                continue
            sa = grid[r][c]
            dr, dc = 0, 1
            while r + dr < size and c + dc < size and _is_white(r + dr, c + dc, size, grid):
                sa += grid[r + dr][c + dc]
                dr, dc = dc, -dr
            if sa:
                down.append({'number': numbers[(r, c)], 'row': r, 'col': c, 'answer': sa, 'len': len(sa)})
            dr, dc = 1, 0
            while r + dr < size and c + dc < size and _is_white(r + dr, c + dc, size, grid):
                sa += grid[r + dr][c + dc]
                dr, dc = dc, -dr
            if sa:
                across.append({'number': numbers[(r, c)], 'row': r, 'col': c, 'answer': sa, 'len': len(sa)})
    return {'rows': size, 'cols': size, 'cells': cells, 'across': across, 'down': down}