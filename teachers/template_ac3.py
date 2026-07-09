"""Fixed-grid crossword generator: pre-verified grid templates + AC-3 fill (family 4).

For LARGE grids (11x11+) random black-square construction almost never yields a
fillable grid (empirically ~9% of random 180-symmetric structures can be filled),
so a generate-and-hope loop burns its whole budget. This generator instead draws
from a small library of PRE-VERIFIED fillable grid templates -- symmetric, fully
interlocked (all cells checked), every slot length >= 3 -- and solves the dictionary
fill as a CSP with AC-3 arc consistency, seating theme/topic words in the long slots
first to maximize vocabulary coverage.

Only the empty grid GEOMETRY is fixed; every answer comes from `word_source` at run
time (nothing about the words is hardcoded). Self-contained (stdlib + random),
signature generate_crossword(topic, word_source, size).
"""

import random
import time

# Pre-verified fillable templates: black-square coordinates for a 180-symmetric,
# fully-interlocked grid. Verified by AC-3-filling each from a large word list.
_TEMPLATES = {
    11: [
        [[0, 3], [0, 7], [1, 7], [3, 0], [3, 1], [3, 5], [3, 6], [3, 10], [4, 3], [5, 3], [5, 7], [6, 7], [7, 0], [7, 4], [7, 5], [7, 9], [7, 10], [9, 3], [10, 3], [10, 7]],
        [[0, 3], [0, 4], [1, 3], [3, 5], [3, 9], [3, 10], [4, 4], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 6], [7, 0], [7, 1], [7, 5], [9, 7], [10, 6], [10, 7]],
        [[0, 3], [0, 7], [3, 4], [3, 5], [3, 10], [4, 3], [4, 4], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 6], [6, 7], [7, 0], [7, 5], [7, 6], [10, 3], [10, 7]],
        [[0, 3], [0, 4], [0, 5], [1, 4], [1, 5], [2, 5], [3, 0], [3, 6], [3, 10], [5, 3], [5, 7], [7, 0], [7, 4], [7, 10], [8, 5], [9, 5], [9, 6], [10, 5], [10, 6], [10, 7]],
        [[0, 6], [0, 7], [3, 0], [3, 1], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [5, 3], [5, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [7, 9], [7, 10], [10, 3], [10, 4]],
        [[0, 3], [0, 7], [1, 3], [1, 7], [3, 4], [4, 0], [4, 4], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 6], [6, 10], [7, 6], [9, 3], [9, 7], [10, 3], [10, 7]],
        [[0, 3], [0, 7], [1, 7], [3, 0], [3, 1], [3, 2], [3, 6], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 4], [7, 8], [7, 9], [7, 10], [9, 3], [10, 3], [10, 7]],
        [[0, 6], [0, 7], [1, 6], [3, 0], [3, 4], [3, 5], [4, 0], [4, 7], [5, 0], [5, 1], [5, 9], [5, 10], [6, 3], [6, 10], [7, 5], [7, 6], [7, 10], [9, 4], [10, 3], [10, 4]],
        [[0, 3], [0, 7], [3, 0], [3, 5], [4, 0], [4, 1], [4, 7], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 3], [6, 9], [6, 10], [7, 5], [7, 10], [10, 3], [10, 7]],
        [[0, 3], [0, 7], [1, 3], [1, 7], [3, 5], [3, 6], [4, 6], [5, 0], [5, 1], [5, 2], [5, 8], [5, 9], [5, 10], [6, 4], [7, 4], [7, 5], [9, 3], [9, 7], [10, 3], [10, 7]],
        [[0, 3], [0, 7], [3, 0], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 7], [5, 3], [5, 7], [6, 3], [6, 7], [6, 8], [6, 9], [6, 10], [7, 6], [7, 10], [10, 3], [10, 7]],
        [[0, 3], [0, 4], [1, 4], [2, 4], [3, 0], [3, 1], [3, 5], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 5], [7, 9], [7, 10], [8, 6], [9, 6], [10, 6], [10, 7]],
        [[0, 3], [0, 7], [1, 7], [3, 4], [3, 10], [4, 3], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [6, 7], [7, 0], [7, 6], [9, 3], [10, 3], [10, 7]],
        [[0, 3], [0, 7], [1, 3], [1, 7], [2, 3], [3, 0], [3, 4], [3, 9], [3, 10], [4, 4], [6, 6], [7, 0], [7, 1], [7, 6], [7, 10], [8, 7], [9, 3], [9, 7], [10, 3], [10, 7]],
        [[0, 6], [0, 7], [1, 6], [2, 6], [3, 3], [3, 4], [4, 3], [4, 4], [5, 0], [5, 1], [5, 9], [5, 10], [6, 6], [6, 7], [7, 6], [7, 7], [8, 4], [9, 4], [10, 3], [10, 4]],
        [[0, 3], [0, 4], [1, 4], [3, 0], [3, 8], [3, 9], [3, 10], [4, 6], [4, 7], [5, 3], [5, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 2], [7, 10], [9, 6], [10, 6], [10, 7]],
    ],
}


def _split_source(word_source):
    """Return (theme_words, fill_words). Accepts the two-tier {theme,fill} dict or a flat list."""
    if isinstance(word_source, dict):
        theme = [str(w).upper() for w in word_source.get("theme", [])]
        fill = [str(w).upper() for w in word_source.get("fill", [])]
        return theme, fill
    return [], [str(w).upper() for w in word_source]


def _index_by_length(words):
    idx = {}
    for w in words:
        w = str(w).upper()
        if w.isalpha():
            idx.setdefault(len(w), []).append(w)
    return idx


def _runs(white, size):
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(size):
            for c in range(size):
                if (r, c) not in white or (r - dr, c - dc) in white:
                    continue
                cells, rr, cc = [], r, c
                while (rr, cc) in white:
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                out.append(cells)
    return out


def _slots_and_crossings(white, size):
    slots = [{"cells": cells, "len": len(cells)} for cells in _runs(white, size)]
    cell_to_slots = {}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            cell_to_slots.setdefault(cell, []).append(i)
    return slots, cell_to_slots


def _fill(slots, idx, rng, theme_set=None, budget=30000, deadline=None):
    """CSP fill with AC-3 maintained through search. When theme_set is given, seats
    theme words into the longest slots first. Returns {slot_index: word} or None."""
    n = len(slots)
    dom = {si: set(idx.get(slots[si]["len"], [])) for si in range(n)}
    if any(not d for d in dom.values()):
        return None

    cellmap = {}
    for si, s in enumerate(slots):
        for pos, cell in enumerate(s["cells"]):
            cellmap.setdefault(cell, []).append((si, pos))
    neighbors = {si: [] for si in range(n)}
    for lst in cellmap.values():
        for (a, pa) in lst:
            for (b, pb) in lst:
                if a != b:
                    neighbors[a].append((b, pa, pb))

    def revise(d, x, y, px, py):
        avail = {w[py] for w in d[y]}
        new = {w for w in d[x] if w[px] in avail}
        if len(new) != len(d[x]):
            d[x] = new
            return True
        return False

    def ac3(d, queue):
        while queue:
            if deadline is not None and time.perf_counter() > deadline:
                return False
            x, y, px, py = queue.pop()
            if revise(d, x, y, px, py):
                if not d[x]:
                    return False
                for (z, pxz, pz) in neighbors[x]:
                    if z != y:
                        queue.append((z, x, pz, pxz))
        return True

    if not ac3(dom, [(a, b, pa, pb) for a in range(n) for (b, pa, pb) in neighbors[a]]):
        return None

    used, assign, steps = set(), {}, [0]

    def select(d):
        unassigned = [s for s in range(n) if s not in assign]
        if theme_set:
            # longest slots first (seat long theme words), tie-break smallest domain
            return min(unassigned, key=lambda s: (-slots[s]["len"], len(d[s])))
        return min(unassigned, key=lambda s: len(d[s]))  # MRV

    def order(d, si):
        cands = [w for w in d[si] if w not in used]
        rng.shuffle(cands)
        if theme_set:
            cands.sort(key=lambda w: w not in theme_set)  # theme words first (stable)
        return cands

    def bt(d):
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return None
        steps[0] += 1
        if len(assign) == n:
            return dict(assign)
        si = select(d)
        for w in order(d, si):
            if deadline is not None and time.perf_counter() > deadline:
                return None  # check BEFORE the expensive domain copy, not just inside ac3()
            nd = {k: set(v) for k, v in d.items()}
            nd[si] = {w}
            queue = [(b, si, pb, ps) for (b, ps, pb) in neighbors[si]]
            if ac3(nd, queue):
                assign[si] = w
                used.add(w)
                r = bt(nd)
                if r is not None:
                    return r
                del assign[si]
                used.discard(w)
        return None

    return bt(dom)


def _build_layout(white, size, slots, assignment):
    grid = {}
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            grid[(r, c)] = word[pos]

    def is_white(r, c):
        return (r, c) in grid

    numbers, num = {}, 0
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if not is_white(r, c):
                continue
            sa = (c == 0 or not is_white(r, c - 1)) and (c + 1 < size and is_white(r, c + 1))
            sd = (r == 0 or not is_white(r - 1, c)) and (r + 1 < size and is_white(r + 1, c))
            if sa or sd:
                num += 1
                numbers[(r, c)] = num
            if sa:
                w, cc = "", c
                while cc < size and is_white(r, cc):
                    w += grid[(r, cc)]
                    cc += 1
                across.append({"number": num, "row": r, "col": c, "answer": w, "len": len(w)})
            if sd:
                w, rr = "", r
                while rr < size and is_white(rr, c):
                    w += grid[(rr, c)]
                    rr += 1
                down.append({"number": num, "row": r, "col": c, "answer": w, "len": len(w)})
    cells = []
    for (r, c), ch in sorted(grid.items()):
        cell = {"r": r, "c": c, "letter": ch}
        if (r, c) in numbers:
            cell["number"] = numbers[(r, c)]
        cells.append(cell)
    return {"rows": size, "cols": size, "cells": cells, "across": across, "down": down}


def generate_crossword(topic, word_source, size):
    deadline = time.perf_counter() + 8.0
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    idx = _index_by_length(theme + fill)
    theme_set = set(theme)
    empty = {"rows": size, "cols": size, "cells": [], "across": [], "down": []}

    templates = _TEMPLATES.get(size)
    if not templates:
        return empty

    full = {(r, c) for r in range(size) for c in range(size)}
    whites = []
    for ti in range(len(templates)):
        white = full - {(r, c) for (r, c) in templates[ti]}
        whites.append((white, _slots_and_crossings(white, size)[0]))
    rng.shuffle(whites)

    # A fillable template usually solves in well under a second; the occasional
    # failure is an unlucky deep search, not an unsolvable grid. So use a short
    # per-attempt budget and RESTART with fresh randomization, cycling templates.
    #
    # VALIDITY FIRST: guarantee a valid grid with a plain fill before spending any
    # budget chasing coverage -- otherwise a hard theme-first search can eat the
    # whole deadline and return nothing (the failure mode seen under CPU load).
    fallback = None
    while fallback is None and time.perf_counter() < deadline:
        for white, slots in whites:
            if time.perf_counter() > deadline:
                break
            a = _fill(slots, idx, rng, theme_set=None,
                      deadline=min(deadline, time.perf_counter() + 1.8))
            if a and len(a) == len(slots):
                fallback = _build_layout(white, size, slots, a)
                break
        rng.shuffle(whites)

    # COVERAGE (bonus): with the remaining budget, try theme-first to seat SAT
    # vocabulary in the long slots. Prefer it if it solves; else keep the fallback.
    while time.perf_counter() < deadline:
        for white, slots in whites:
            if time.perf_counter() > deadline:
                break
            a = _fill(slots, idx, rng, theme_set=theme_set,
                      deadline=min(deadline, time.perf_counter() + 1.8))
            if a and len(a) == len(slots):
                return _build_layout(white, size, slots, a)
        rng.shuffle(whites)
    return fallback if fallback is not None else empty
