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
# word_source is provided at runtime (a list, or a {"theme","fill"} dict of
# prioritized vocabulary + fill words); NEVER invent or hardcode words. Choose the
# construction and fill strategy (e.g. CSP backtracking with MRV + forward checking,
# AC-3 / maintained arc consistency, a (length,position,letter) pattern index, beam
# search, theme-first ordering to maximize vocabulary). Prefer packing vocabulary
# words where the crossings allow.

"""gen3 fusion: beam_search (family 3: beam, no backtracking) + THEME-WEIGHTED
scoring. gen1/gen2 only added theme-first to backtracking fillers; the beam scorer
was theme-blind. Here candidates are considered theme-first and each beam state is
scored by neighbor-openness PLUS a dominant bonus for placing SAT words -- so the
beam packs vocabulary while its forward-check still guarantees feasibility.

Self-contained; generate_crossword(topic, word_source, size); never hardcodes words.
"""

import random
import time

_THEME_BONUS = 10000   # >> typical openness score -> theme primary, openness tiebreak


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
        return list(by_len.get(slot["len"], []))
    sets = []
    for pos, ch in fixed:
        s = pat.get((slot["len"], pos, ch))
        if not s:
            return []
        sets.append(s)
    sets.sort(key=len)
    result = set(sets[0])
    for s in sets[1:]:
        result &= s
        if not result:
            break
    return list(result)


def _fill(slots, cell_to_slots, idx, rng, theme_set, beam_width=30, cand_per_slot=14,
          budget=200000, deadline=None):
    by_len = idx
    pat = _build_pattern_index(idx)
    n = len(slots)
    neighbors = {si: set() for si in range(n)}
    for si, s in enumerate(slots):
        for cell in s["cells"]:
            for sj in cell_to_slots[cell]:
                if sj != si:
                    neighbors[si].add(sj)

    def n_cross(si):
        return sum(1 for cell in slots[si]["cells"] if len(cell_to_slots[cell]) > 1)

    order = sorted(range(n), key=lambda si: (-n_cross(si), -slots[si]["len"]))

    states = [({}, {}, set())]
    steps = [0]
    for si in order:
        slot = slots[si]
        scored = []
        for (assign, letters, used) in states:
            cands = [w for w in _pool(slot, letters, pat, by_len) if w not in used]
            rng.shuffle(cands)
            cands.sort(key=lambda w: w not in theme_set)   # consider theme words first
            for w in cands[:cand_per_slot]:
                steps[0] += 1
                if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
                    break
                nl = dict(letters)
                ok = True
                for p, cell in enumerate(slot["cells"]):
                    ex = nl.get(cell)
                    if ex is not None and ex != w[p]:
                        ok = False
                        break
                    nl[cell] = w[p]
                if not ok:
                    continue
                dead, score = False, 0
                for rj in neighbors[si]:
                    if rj in assign:
                        continue
                    cnt = sum(1 for x in _pool(slots[rj], nl, pat, by_len) if x not in used and x != w)
                    if cnt == 0:
                        dead = True
                        break
                    score += cnt
                if dead:
                    continue
                if w in theme_set:                       # theme-weighted: prefer SAT words
                    score += _THEME_BONUS
                na = dict(assign)
                na[si] = w
                nu = set(used)
                nu.add(w)
                scored.append((score, na, nl, nu))
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        states = [(a, l, u) for (_, a, l, u) in scored[:beam_width]]

    for assign, _, _ in states:
        if len(assign) == n:
            return assign
    return None


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


def generate_crossword(topic: str, word_source, size: int) -> dict:
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
        assignment = _fill(slots, cell_to_slots, idx, rng, theme_set, deadline=deadline)
        if assignment and len(assignment) == len(slots):
            return _build_layout(white, size, slots, assignment)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
