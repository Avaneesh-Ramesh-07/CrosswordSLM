"""Teacher generator: VOCAB-FIRST fill (maximizes coverage).

The education behavior we want to teach: actually PLACE the target vocabulary,
not just fill a legal grid. This generator receives a structured word source
    word_source = {"theme": [SAT words to feature], "fill": [common connectors]}
and, at every slot, tries THEME words before fill words. Backtracking keeps the
grid valid; the theme-first ordering packs in as many vocabulary words as the
crossings allow -> high `coverage`. A naive filler ignores the distinction and
scores near-zero coverage; that gap is the behavior the fine-tune should learn.

Self-contained (stdlib + random). Accepts word_source as the dict above, or a
flat list (treated as all-fill) for backward compatibility.
"""

import random

# EVOLVE-BLOCK-START
import time


def _split_source(word_source):
    if isinstance(word_source, dict):
        theme = [str(w).upper() for w in word_source.get("theme", []) if str(w).isalpha()]
        fill = [str(w).upper() for w in word_source.get("fill", []) if str(w).isalpha()]
    else:
        theme, fill = [], [str(w).upper() for w in word_source if str(w).isalpha()]
    return theme, fill


def _index_by_length(words):
    idx = {}
    for w in words:
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


def _build_pattern_index(idx_by_len):
    pat = {}
    for length, words in idx_by_len.items():
        for w in words:
            for pos, ch in enumerate(w):
                pat.setdefault((length, pos, ch), set()).add(w)
    return pat


def _pool(slot, letters, pat, by_len):
    fixed = [(pos, letters[cell]) for pos, cell in enumerate(slot["cells"]) if cell in letters]
    if not fixed:
        return set(by_len.get(slot["len"], []))
    sets = []
    for pos, ch in fixed:
        s = pat.get((slot["len"], pos, ch))
        if not s:
            return set()
        sets.append(s)
    sets.sort(key=len)
    result = set(sets[0])
    for s in sets[1:]:
        result &= s
        if not result:
            break
    return result


def _fill(slots, cell_to_slots, idx, theme_set, rng, budget=1500, deadline=None):
    by_len = idx
    pat = _build_pattern_index(idx)
    letters, assignment, used = {}, {}, set()
    steps = [0]
    neigh = []
    for si, s in enumerate(slots):
        nb = set()
        for cell in s["cells"]:
            for other in cell_to_slots[cell]:
                if other != si:
                    nb.add(other)
        neigh.append(nb)

    def dom_size(si):
        return len(_pool(slots[si], letters, pat, by_len))

    def backtrack():
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return False
        steps[0] += 1
        if len(assignment) == len(slots):
            return True
        unassigned = [si for si in range(len(slots)) if si not in assignment]
        for si in unassigned:
            if dom_size(si) == 0:
                return False  # dead end -> backtrack
        # Seat theme words where they FIT: longest slots first (SAT words are long),
        # tie-break by most-constrained. This packs vocabulary into the long entries
        # before short-slot fill locks the crossings.
        best_si = max(unassigned, key=lambda si: (slots[si]["len"], -dom_size(si)))
        pool = [w for w in _pool(slots[best_si], letters, pat, by_len) if w not in used]
        # THEME-FIRST: try vocabulary words before fill words for this slot.
        theme_c = [w for w in pool if w in theme_set]
        other_c = [w for w in pool if w not in theme_set]
        rng.shuffle(theme_c)
        rng.shuffle(other_c)
        for word in theme_c + other_c:
            changed = []
            for pos, cell in enumerate(slots[best_si]["cells"]):
                if cell not in letters:
                    letters[cell] = word[pos]
                    changed.append(cell)
            assignment[best_si] = word
            used.add(word)
            dead = any(nb not in assignment and dom_size(nb) == 0 for nb in neigh[best_si])
            if not dead and backtrack():
                return True
            del assignment[best_si]
            used.discard(word)
            for cell in changed:
                del letters[cell]
        return False

    return dict(assignment) if backtrack() else None


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
    deadline = time.perf_counter() + 4.0
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    idx = _index_by_length(theme + fill)
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    for _ in range(200):
        if time.perf_counter() > deadline:
            break
        white = _make_structure(size, rng)
        slots, cell_to_slots = _slots_and_crossings(white, size)
        assignment = _fill(slots, cell_to_slots, idx, theme_set, rng, budget=1500, deadline=deadline)
        if assignment and len(assignment) == len(slots):
            return _build_layout(white, size, slots, assignment)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}

# EVOLVE-BLOCK-END
