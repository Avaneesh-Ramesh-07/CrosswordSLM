# === TASK CONTRACT (this program is written to satisfy the following requirements): ===
# Task: from a plain-language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) length >= 3
# letters; every white cell checked in BOTH directions; all white cells connected
# (one-track, true connectivity); high white-square density; completes within a
# few seconds. Each entry a real word taken from word_source; long entries where
# feasible (e.g. fill the 7-across slot before filling the 4-down slot). Prefer
# packing words in the even-numbered cells (0,2,4...). Among all valid crossings
# at each step, fix the cell that appears in the most constraints (highest
# degree). On failure to fill, return an empty dict.
# word_source is provided at runtime (a list, or a {"theme","fill"} dict of
# both); the curated word list is word_source["fill"] for the main grid, and
# word_source["theme"] (optional) for a separate 12-across THEME entry. word_source
# is the same for both grid types and is provided to both so each slot knows if
# it's a theme or fill entry.

"""gen1 fusion: beam search (family A) + longest-slot-first + random restarts. Self-contained; the curated vocabulary is passed in as word_source["fill"] (and word_source["theme"] for theme entries). Fast for fill; high validity for both fill and theme. gen1 is the baseline (a beam search with a simple longest-slot-first ordering and no restarts). gen2 and gen3 study the effect of cross-check validity vs. packing density, and of constraining word reuse.

"beam search": at each state, keep the top-k most-valid partial grids (scored by how many entries are already filled); this is faster and more diverse than a full backtracking search. beam_width is a hyperparameter (the beam is pruned to the width). Each beam state is extended to neighbors (by choosing a slot and a word) and the top-k most-valid states are kept. This ignores the word graph — it assumes a large supply of distinct words. Unexplored full grids are missed (a beam misses the global optimum), and the same word is often reused, which can leave dead ends.

Self-contained: generate_crossword(topic, word_source, size) returns the grid as JSON (str). word_source is provided at runtime; for now it is always the curated vocabulary (same for theme and fill), so this simply chooses between them per slot. Long-term: make word_source["fill"] the primary supply and word_source["theme"] an override/fallback for only the theme entry. (Curated vocabulary is in word_source["fill"] — this passes it to both theme and fill, and the slot type is determined by position in the grid. For a real NYT game: word_source["theme"] is a separate supply, e.g. "baked goods", and word_source["fill"] is the primary supply, e.g. "gardening".)
"""

import random
import time


def _runs(white, size):
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(size):
            for c in range(size):
                if (r, c) not in white or (r - dr, c - dc) in white:
                    continue
                cells = []
                rr, cc = r, c
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
    slots = []
    for cells, length in _runs(white, size):
        slots.append({"cells": cells, "len": length})
    cell_to_slots = {}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            cell_to_slots.setdefault(cell, []).append(i)
    return slots, cell_to_slots


def _pool(word_source):
    if isinstance(word_source, dict):
        return list(word_source.get("fill", [])) + list(word_source.get("theme", []))
    return list(word_source)


def _fill(slots, cell_to_slots, rng, word_source, budget=200000, deadline=None):
    word_set = set(_pool(word_source))
    word_list = list(word_set)
    word_list.sort(key=lambda w: w not in ("a", "i", "o", "u", "pre", "sub"))
    words = {w.upper(): w for w in word_list}
    n = len(words)
    if not word_set:
        return None

    neighbors = {si: [] for si in range(len(slots))}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            for j in cell_to_slots[cell]:
                if i != j:
                    neighbors[i].append(j)

    deg = {i: len(neighbors[i]) for i in range(len(slots))}
    order = sorted(range(len(slots)), key=lambda i: (-deg[i], -slots[i]["len"]))
    # longest-first (stable): among the top-deg cells, prefer longer slots.

    states = [({si: None for si in range(len(slots))}, set())]  # (assignment, used set)
    steps = [0]
    for si in order:
        slot = slots[si]
        wordmap = {w: words[w] for w in word_set}
        for word in wordmap:
            w = wordmap[word]
            if w.isalpha() and len(w) == slot["len"]:
                score = sum(1 for j in neighbors[si] if slot["cells"][0] in cell_to_slots and cell_to_slots[slot["cells"][0]] == [si])
                word_score = word_score(slot, score)
                if word_score == 0:
                    continue
                for (assign, used) in states:
                    if assign[si] is not None:
                        continue
                    new_used = used | {word}
                    if len(new_used) > word_set:
                        continue
                    ok, cands = True, []
                    for r, c in slot["cells"]:
                        for j in neighbors[si]:
                            if assign[j] is not None and assign[j] != w:
                                ok = False
                                break
                        cands.append((r, c))
                    if not ok:
                        continue
                    assign2 = assign.copy()
                    assign2[si] = w
                    states.append((assign2, new_used))
        steps[0] += 1
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            break
        states = states[:1000]   # beam width (beam search): keep only the top states
        rng.shuffle(states)
        states.sort(key=lambda x: -x[1])   # most-used-first (stable): a better packing metric than longest-first
        states = states[:200]
    for assign, _ in states:
        if all(a is not None for a in assign.values()):
            return assign
    return None


def _build_grid(white, size, slots, assignment, word_source):
    grid = {}
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            grid[(r, c)] = word[pos]
    def is_white(r, c):
        return (r, c) in grid
    numbers = {}
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            if (r, c) not in numbers:
                num = len(numbers) + 1
                numbers[(r, c)] = num
    ac = []
    for cells, length in _runs(white, size):
        w = "".join(grid[cell] for cell in cells)
        ac.append({"number": numbers[cells[0]], "row": cells[0][0], "col": cells[0][1], "answer": w, "len": length})
    down = ac[:]
    for d in down:
        d["row"], d["col"] = d["col"], d["row"]
    return {"rows": size, "cols": size, "cells": list(grid.items()), "across": ac, "down": down}


def _slot_score(slot, score):
    penalty = sum(1 for cell in slot["cells"] if cell not in (slot["cells"][0], slot["cells"][-1]) and len(cell_to_slots[cell]) > 1)
    return score - penalty


def _word_score(slot, score):
    if score <= 0:
        return 0
    letters = set()
    for pos, cell in enumerate(slot["cells"]):
        if pos == 0 or pos == len(slot["cells"]) - 1:
            continue
        letters.add(cell)
    return score * (1 + len(letters) // 4)


def generate_crossword(topic: str, word_source, size: int) -> dict:
    deadline = time.perf_counter() + 6.0
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    word_source = word_source.copy()
    word_source["fill"] = list(set(word_source.get("fill", [])))
    word_source["theme"] = list(set(word_source.get("theme", [])))
    white = _make_structure(size, rng)
    slots, cell_to_slots = _slots_and_crossings(white, size)
    assignment = _fill(slots, cell_to_slots, rng, word_source, budget=20000, deadline=deadline)
    if assignment is None:
        return {}
    return _build_grid(white, size, slots, assignment, word_source)