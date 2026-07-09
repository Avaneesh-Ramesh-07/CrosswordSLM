"""gen3 fusion: csp_ac3 (AC-3/MAC + MRV) + true LEAST-CONSTRAINING-VALUE ordering
(unexplored -- no seed did classic LCV) with theme-first as the primary key. Among
a slot's candidates it prefers SAT words (coverage), and orders by how many options
each leaves open for crossing slots (LCV) so the maintained-arc-consistency search
dead-ends less often.

Self-contained; generate_crossword(topic, word_source, size); never hardcodes words.
"""

import random
import time

_LCV_WINDOW = 30   # compute LCV only for the top candidates (bounds cost)


def _split_source(word_source):
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
                if (r, c) not in white or (r - dr, c - dc) in white:
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
            if _structure_ok(full - (blacks | {(r, c), partner}), size, min_len):
                blacks |= {(r, c), partner}
        white = full - blacks
        if _structure_ok(white, size, min_len):
            return white
    return full


def _slots_and_crossings(white, size):
    slots = [{"cells": cells, "len": length} for cells, length in _runs(white, size)]
    cell_to_slots = {}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            cell_to_slots.setdefault(cell, []).append(i)
    return slots, cell_to_slots


def _fill(slots, cell_to_slots, idx, rng, theme_set, budget=8000, deadline=None):
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

    def bt(d):
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return None
        steps[0] += 1
        if len(assign) == n:
            return dict(assign)
        si = min((s for s in range(n) if s not in assign), key=lambda s: len(d[s]))  # MRV
        cands = [w for w in d[si] if w not in used]
        rng.shuffle(cands)
        cands.sort(key=lambda w: w not in theme_set)   # theme-first (primary)

        def lcv(w):  # least-constraining-value: options this word leaves for neighbors
            total = 0
            for (b, pa, pb) in neighbors[si]:
                if b in assign:
                    continue
                ch = w[pa]
                total += sum(1 for x in d[b] if x[pb] == ch)
            return total

        head = cands[:_LCV_WINDOW]
        head.sort(key=lambda w: (w not in theme_set, -lcv(w)))   # theme-first, then most-open
        cands = head + cands[_LCV_WINDOW:]

        for w in cands:
            if deadline is not None and time.perf_counter() > deadline:
                return None
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
    deadline = time.perf_counter() + 6.0
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    idx = _index_by_length(theme + fill)
    for _ in range(200):
        if time.perf_counter() > deadline:
            break
        white = _make_structure(size, rng)
        slots, cell_to_slots = _slots_and_crossings(white, size)
        a = _fill(slots, cell_to_slots, idx, rng, theme_set,
                  deadline=min(deadline, time.perf_counter() + 2.0))
        if a and len(a) == len(slots):
            return _build_layout(white, size, slots, a)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
