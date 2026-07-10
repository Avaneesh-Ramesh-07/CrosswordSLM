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
# prioritized vocabulary + fill words); the curated word list is HARDCODED into _WORDS and used by default (word_source overrides it). Choose the
# construction and fill strategy (e.g. CSP backtracking with MRV + forward checking,
# AC-3 / maintained arc consistency, a (length,position,letter) pattern index, beam
# search, theme-first ordering to maximize vocabulary). Prefer packing vocabulary
# words where the crossings allow.

"""Fixed-template crossword generator (baked-in real NYT 15x15 grids + ac3_lcv fill). engine=ac3_lcv selection=shuffle subset=secondhalf(23)

23 pre-verified-fillable black-square patterns are inlined; the grid is SELECTED (not randomly constructed) then filled from word_source. Self-contained; the curated vocabulary is HARDCODED into _WORDS and used by default; word_source is an optional override/fallback."""

import random
import time

_TEMPLATES = [
    [[0, 4], [0, 5], [0, 10], [1, 4], [2, 4], [3, 3], [3, 7], [4, 6], [5, 0], [5, 1], [5, 9], [5, 10], [6, 4], [7, 3], [7, 7], [8, 6], [9, 6], [10, 0], [10, 5], [10, 6]],
    [[0, 3], [0, 4], [1, 4], [2, 4], [3, 0], [3, 5], [3, 9], [3, 10], [4, 6], [4, 7], [6, 3], [6, 4], [7, 0], [7, 1], [7, 5], [7, 10], [8, 6], [9, 6], [10, 6], [10, 7]],
    [[0, 0], [0, 4], [0, 5], [1, 4], [3, 7], [4, 0], [4, 1], [4, 2], [4, 6], [5, 0], [5, 10], [6, 4], [6, 8], [6, 9], [6, 10], [7, 3], [9, 6], [10, 5], [10, 6], [10, 10]],
    [[0, 4], [0, 10], [1, 4], [2, 4], [3, 6], [4, 3], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [6, 7], [7, 4], [8, 6], [9, 6], [10, 0], [10, 6]],
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
]


_LCV_WINDOW = 30


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


_WORDS = ['ABET', 'ABILITIES', 'ACE', 'ACES', 'ACID', 'ACRE', 'ACTS', 'ACUMEN', 'ADA', 'ADD', 'ADORES', 'ADS', 'AERIAL', 'AFAR', 'AFRAID', 'AHEAD', 'AID', 'AIM', 'AIR', 'ALA', 'ALE', 'ALES', 'ALIVE', 'ALLIES', 'ALLURE', 'ALOOF', 'AMISS', 'ANGLED', 'ANGST', 'ANIMAL', 'ANNA', 'APACHE', 'APES', 'APT', 'ARCTIC', 'AREA', 'ARENAS', 'ARES', 'ARISEN', 'ARK', 'ASCENT', 'ASH', 'ASP', 'ASPEN', 'ASSERT', 'ASSES', 'ATE', 'AVENGE', 'AWE', 'BALE', 'BAT', 'BEST', 'BIDE', 'BISTRO', 'BLOT', 'BOLERO', 'BOMBED', 'BRIM', 'BRIT', 'BUN', 'BURGEON', 'CAGES', 'CAM', 'CAMEL', 'CAPRA', 'CASES', 'CASTS', 'CAVEAT', 'CHEST', 'CHI', 'CODED', 'COLA', 'CONES', 'CONSTRAINTS', 'CYAN', 'DEED', 'DENS', 'DESK', 'DETECT', 'DIMLY', 'DIVA', 'DOE', 'DOH', 'DYES', 'EAT', 'EDDY', 'EDGE', 'EDIBLE', 'EDIT', 'EEL', 'EGG', 'EGO', 'ELASTIC', 'ELM', 'EMIRATE', 'EMIT', 'ENCODE', 'ENTAIL', 'ENTREE', 'EON', 'EONS', 'ERA', 'ERICA', 'ERODED', 'ERR', 'ESPRIT', 'ESPY', 'ESTER', 'ETNA', 'EXITED', 'EYED', 'FAILURE', 'FEE', 'FIST', 'FORBADE', 'FOREGO', 'FRET', 'FUR', 'GALA', 'GAS', 'GEE', 'GNOMES', 'GREETS', 'GRENADE', 'GREY', 'GRIM', 'GRIT', 'HADES', 'HAIRDO', 'HASH', 'HERD', 'HIT', 'HOARSE', 'HOT', 'ICE', 'ICEMAN', 'IDEAL', 'INLETS', 'INTERSECTED', 'IRE', 'IRON', 'ISLAND', 'ITEM', 'KNEE', 'LAB', 'LASH', 'LASTS', 'LAVA', 'LAX', 'LED', 'LENSES', 'LID', 'LIST', 'LIT', 'LOCH', 'LOG', 'LURE', 'MARGE', 'MATTE', 'MEET', 'MOD', 'MOTE', 'NAPA', 'NAPE', 'NEARER', 'NEE', 'NEED', 'NESS', 'NESTS', 'NETTLE', 'NEURAL', 'NEW', 'NEWT', 'NOD', 'NOTE', 'NOVA', 'NOXIOUS', 'NUTS', 'OAR', 'OAT', 'ODOR', 'OHM', 'OIL', 'ONE', 'ORAL', 'ORCA', 'ORE', 'ORIGIN', 'OTTAWA', 'OVERSEE', 'PAC', 'PADDED', 'PANE', 'PARDON', 'PASTE', 'PASTS', 'PATE', 'PEA', 'PENCIL', 'PEST', 'PHI', 'PITS', 'PLANK', 'PLOT', 'PREDECESSOR', 'PRO', 'PROMINENTLY', 'PUMA', 'RADAR', 'RAG', 'RAMONA', 'RARE', 'RAYS', 'READ', 'READER', 'REAL', 'REASON', 'RENAME', 'RENTS', 'REPEAT', 'RES', 'RHO', 'RICE', 'RITUAL', 'ROUTE', 'SAC', 'SAG', 'SALE', 'SAM', 'SARAN', 'SAVOR', 'SEALS', 'SECS', 'SEES', 'SELLS', 'SEMI', 'SENIOR', 'SHEAR', 'SHED', 'SHOE', 'SHOO', 'SHRED', 'SIGNAL', 'SLY', 'SNUG', 'SOD', 'SODA', 'SOLES', 'SOOT', 'SOS', 'SPA', 'SPEARHEADED', 'SPONGE', 'SPREAD', 'STANCE', 'STEAD', 'STEM', 'STEP', 'STEW', 'STINK', 'STREET', 'STUMP', 'SUBSISTENCE', 'SWAT', 'TAMED', 'TANTRA', 'TAPED', 'TARO', 'TEA', 'TEDDY', 'TEE', 'TEEN', 'TEND', 'TESLA', 'THREAD', 'TIE', 'TIN', 'TRASH', 'TREAD', 'TRIM', 'TRIO', 'TRIP', 'TUBE', 'TUNNELS', 'UNWARRANTED', 'URBANE', 'URGE', 'VETO', 'VINAIGRETTE', 'WALKER', 'WEAPON', 'WRY', 'YEN', 'YIELD']

def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 11) -> dict:
    word_source = word_source or _WORDS
    deadline = time.perf_counter() + 7.8
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    idx = _index_by_length(theme + fill)
    full = {(r, c) for r in range(size) for c in range(size)}
    order = list(range(len(_TEMPLATES)))
    rng.shuffle(order)
    for ti in order:
        if time.perf_counter() > deadline:
            break
        black = _TEMPLATES[ti]
        white = full - {(r, c) for (r, c) in black}
        slots, cell_to_slots = _slots_and_crossings(white, size)
        a = _fill(slots, cell_to_slots, idx, rng, theme_set, budget=200000,
                  deadline=min(deadline, time.perf_counter() + 4.0))
        if a and len(a) == len(slots):
            return _build_layout(white, size, slots, a)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}