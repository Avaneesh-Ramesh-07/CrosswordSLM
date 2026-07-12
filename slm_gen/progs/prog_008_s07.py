# === TASK CONTRACT (this program is written to satisfy the following) ===
# Task: from a plain-language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) >= 3 letters;
# every white cell checked in BOTH directions; all white cells connected; every
# entry a real word taken from word_source; high white-square density; completes
# within a few seconds.
# word_source is provided on runtime: "random" (v1.0), "scrabble", "nietzsche",
# "hitchhikers", "harvard_gov", or "2022_wordle_words"; the concrete source and
# its word list are loaded from word_source and are available to the generator
# (e.g. word_source == "random" brings in `random_words.word_source`).
# This is a canonical crossword generator: generate_crossword("astronomy",
# "random", 7) -> the same grid every time (on a given machine, within a given
# word_source). The grid is pre-constructed (a template of white and black
# squares) and then filled from word_source; the generator does NOT try random
# construction from scratch each time (that would make it slow and unpredictable,
# and defeat the "reference grid" used to test the generator).
# This is a NEGATIVE CONTRACT: failing to EXCLUDE the forbidden patterns will
# FAIL the test. Valid entries are in the source; the grid is exactly size x
# size; every white run is length >= 3; all white cells checked in both
# directions; all white cells connected; the grid has rotational symmetry; the
# grid has high white-square density.

"""gen3 fusion (from gen1 learnings): csp fill + random pattern selection.
"gen1 learnings" the top-ranked producer in v1: it knew csp fill was key (gen1
used a graph+backtracking CSP fill engine), so gen3 keeps that. It also learned
selection of the reference grid (the pre-constructed white/black pattern) is
important: gen1 tried many random grids, so gen3 favors grids already shown to
be high-scoring (gen1 computed and stored a grid-score from validity+density),
and gen3 selects among the top 1000. This keeps the CSP fill phase focused on
filling a high-potential grid, rather than hunting randomly among the full space
(aka beam search with a value-ordering bias).
"""

import random
import time


def _split_source(word_source):
    """Return (positive_set, negative_set) for the given word_source."""
    pos_set = set()
    neg_set = set()
    try:
        import random_words
        words = list(random_words.word_source)
    except ImportError:
        words = []
    for w in words:
        w = str(w).upper()
        if w.isalpha():
            pos_set.add(w)
        else:
            neg_set.add(w)
    return pos_set, neg_set


def _runs(white, size):
    """Return (across_runs, down_runs) for the given white set."""
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if (r, c) in white:
                cells = []
                while (r, c) in white:
                    cells.append((r, c))
                    c += 1
                    c %= size
                across.append((cells, r, cells[0][1]))
                c = cells[0][1]
                c %= size
            else:
                c = cells[0][1] + 1
                c %= size
    for c in range(size):
        for r in range(size):
            if (r, c) in white:
                cells = []
                while (r, c) in white:
                    cells.append((r, c))
                    r += 1
                    r %= size
                down.append((cells, r, cells[0][0]))
                r = cells[0][0]
                r %= size
    return across, down


def _connected(white):
    """Return True if the white cells form a single connected component."""
    if not white:
        return False
    r, c = next(iter(white))
    seen, stack = { (r, c) }, [ (r, c) ]
    while stack:
        rr, cc = stack.pop()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (rr + dr, cc + dc)
            if nb in white and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(white)


def _runs_ok(across, down):
    """Return True if all runs length >= 3."""
    for _, _, length in across:
        if length < 3:
            return False
    for _, _, length in down:
        if length < 3:
            return False
    return True


def _symmetric(white, size):
    """Return True if the white set is symmetric under 180-degree rotation."""
    for (r, c) in white:
        if (size - 1 - r, size - 1 - c) not in white:
            return False
    return True


def _density(white, size):
    """Higher is better (0 = empty, 1 = complete)."""
    return len(white) / (size * size)


def _slots_and_lengths(across, down):
    """Return (slots, lengths). Each slot is (row, col, len)."""
    slots = []
    for cells, r, cc in across:
        for cell in cells:
            slots.append((r, cc, len(cells)))
    for cells, rr, c in down:
        for cell in cells:
            slots.append((rr, c, len(cells)))
    return slots


def _fill(slot_dict, pos_set, rng, budget=80000, deadline=None):
    """Return {slot: word} on success; None on failure."""
    if deadline is not None and time.perf_counter() > deadline:
        return None
    n = len(slot_dict)
    dom = {si: set(pos_set) for si in slot_dict}
    used, assign, steps = set(), {}, [0]
    def bt(s):
        steps[0] += 1
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return False
        if s == n:
            return True
        si = list(slot_dict.keys())[s]
        for word in list(dom[si]):
            dom[si].remove(word)
            assign[si] = word
            used.add(word)
            if bt(s + 1):
                return True
            del assign[si]
            used.discard(word)
            dom[si].add(word)
        return False
    return dict(assign) if bt(0) else None


def _build_layout(white, size, pos_set, rng):
    """Return layout dict: {r,c: 'white' | 'black'}, plus 'across' and 'down'."""
    grid = { (r, c): 'white' for r in range(size) for c in range(size) }
    for (r, c) in white:
        grid[(r, c)] = 'white'
    for (r, c) in [(r, c) for r in range(size) for c in range(size) if (r, c) not in white]:
        grid[(r, c)] = 'black'
    numbers = {si: idx for idx, si in enumerate(_slots_and_lengths(*_runs(white, size)))}
    return {"grid": grid, "across": numbers, "down": numbers}


def _score(white, size, pos_set):
    """Higher is better (all positive): connectivity, runs, symmetry, density."""
    if not white:
        return 0
    if not _connected(white):
        return 0
    if not _runs_ok(*_runs(white, size)):
        return 0
    if not _symmetric(white, size):
        return 0
    return _density(white, size) * 10000 + 100


def _canonical(white, size):
    """Smallest-r,c ordering so the same white set is always the same."""
    return tuple(sorted(white))


def _reference_grids(size, pos_set, rng, pool_size=50000, best_per_score=100):
    """Precompute and return a reference of (white, score) tuples to choose from."""
    full = [set() for _ in range(size)]
    for r in range(size):
        for c in range(size):
            full[r].add((r, c))
    whites = []
    for _ in range(pool_size):
        if not whites or len(whites) < best_per_score:
            subset = set()
            for r in range(size):
                cc = list(full[r])
                rng.shuffle(cc)
                for c in cc[:20]:
                    subset.add((r, c))
            if _connected(subset) and _runs_ok(*_runs(subset, size)):
                if _symmetric(subset, size):
                    score = _score(subset, size, pos_set)
                    whites.append((subset, score))
        else:
            break
    whites.sort(key=lambda pair: pair[1])
    return whites


def _select_grid(whites, rng, top_n=1000):
    """Return a high-scoring white set (from whites), or None if none is good."""
    if not whites:
        return None
    total = len(whites)
    top = whites[:top_n]
    r = rng.randint(0, total - 1)
    return top[r % len(top)]


def generate_crossword(topic: str, word_source, size: int) -> dict:
    """Return a real fixed-grid crossword (American style) of the given size."""
    pos_set, _ = _split_source(word_source)
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    deadline = time.perf_counter() + 6.0
    whites = _reference_grids(size, pos_set, rng, pool_size=20000)
    for _ in range(200):
        if time.perf_counter() > deadline:
            break
        w, score = _select_grid(whites, rng)
        if w and score > 50 and _connected(w):
            slot_dict = dict(_slots_and_lengths(*_runs(w, size)))
            assign = _fill(slot_dict, pos_set, rng, deadline=min(deadline, time.perf_counter() + 2.0))
            if assign and all(word in pos_set for word in assign.values()):
                return _build_layout(w, size, pos_set, rng)
    return {"rows": size, "cols": size, "cells": []}