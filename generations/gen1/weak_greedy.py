"""gen1 NEGATIVE control: deliberately weak greedy filler -- no propagation, no
forward-checking, a single structure attempt and a tiny node budget. It fills easy
7x7s sometimes but fails 9x9/11x11, so it seeds the labeled-negative pool (that's
its purpose: a "what NOT to do" example, per the SOAR framing).

Self-contained; generate_crossword(topic, word_source, size).
"""

import random
import time


def _split_source(word_source):
    if isinstance(word_source, dict):
        return ([str(w).upper() for w in word_source.get("theme", [])]
                + [str(w).upper() for w in word_source.get("fill", [])])
    return [str(w).upper() for w in word_source]


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
    # Intentionally weak: ONE structure, greedy first-fit, no look-ahead, tiny budget.
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    words = _split_source(word_source)
    idx = _index_by_length(words)
    white = _make_structure(size, rng)
    slots, cell_to_slots = _slots_and_crossings(white, size)
    grid, used, assign = {}, set(), {}
    budget = [400]
    order = sorted(range(len(slots)), key=lambda si: -slots[si]["len"])
    for si in order:
        placed = False
        cands = list(idx.get(slots[si]["len"], []))
        rng.shuffle(cands)
        for w in cands[:60]:
            budget[0] -= 1
            if budget[0] <= 0:
                break
            if w in used:
                continue
            ok = True
            for pos, cell in enumerate(slots[si]["cells"]):
                if cell in grid and grid[cell] != w[pos]:
                    ok = False
                    break
            if ok:
                for pos, cell in enumerate(slots[si]["cells"]):
                    grid[cell] = w[pos]
                used.add(w)
                assign[si] = w
                placed = True
                break
        if not placed:
            return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
    return _build_layout(white, size, slots, assign)
