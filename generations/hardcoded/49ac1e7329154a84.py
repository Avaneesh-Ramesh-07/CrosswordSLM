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

"""gen1 fusion: AC-3/MAC (csp_ac3) + theme-first + longest-slot-first (vocab_first),
on random symmetric construction. Unexplored combo: strong arc-consistency
look-ahead AND vocabulary-packing coverage, without a template library.

Self-contained; generate_crossword(topic, word_source=None, size). The curated vocabulary is HARDCODED into _WORDS and used by default; word_source is an optional override/fallback.
"""

import random
import time


def _split_source(word_source):
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


def _fill(slots, idx, rng, theme_set, budget=8000, deadline=None):
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
        # longest slot first (seat long theme words), smallest-domain tiebreak
        si = max((s for s in range(n) if s not in assign),
                 key=lambda s: (slots[s]["len"], -len(d[s])))
        cands = [w for w in d[si] if w not in used]
        rng.shuffle(cands)
        cands.sort(key=lambda w: w not in theme_set)   # theme words first (stable)
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


_WORDS = ['ABRIDGE', 'ABUSIVE', 'ACE', 'ACREAGE', 'ACT', 'ADA', 'ADD', 'ADMIRER', 'ADO', 'AFT', 'AGE', 'ALIBI', 'ALTERNATE', 'ANALOGOUS', 'ANI', 'ANT', 'ANVIL', 'APE', 'APT', 'ARC', 'ARE', 'ARES', 'ARID', 'ARM', 'ARRANGE', 'ASIDE', 'ASS', 'BACH', 'BAT', 'BED', 'BEE', 'BEN', 'BOOTLEG', 'BUMP', 'CAB', 'CANTO', 'CAP', 'CAR', 'CITE', 'COO', 'COUP', 'CURSIVE', 'DAB', 'DAM', 'DEN', 'DEPLETE', 'DERIVED', 'DEVELOP', 'DEW', 'DIAL', 'DIVERSE', 'DIVULGE', 'DOH', 'DOLL', 'DUNK', 'EAR', 'EEL', 'EGG', 'ELK', 'ELM', 'ELS', 'EMIT', 'END', 'ENDEMIC', 'ENTERED', 'EON', 'EPICURE', 'EPISODE', 'ERA', 'ERG', 'ERR', 'ESP', 'ETA', 'EURO', 'EVE', 'EYE', 'FLU', 'FOB', 'FRY', 'GAG', 'GAGS', 'GEE', 'GEM', 'GENIE', 'GOO', 'GRANGER', 'HEED', 'ICE', 'ICY', 'ILL', 'IMPLEMENT', 'INS', 'IRE', 'IRIS', 'ISOLATE', 'KNEE', 'LAP', 'LED', 'LEI', 'LID', 'LIRA', 'LOP', 'LYE', 'MAD', 'MALARIA', 'MAP', 'MAR', 'MAS', 'METHODS', 'MILO', 'MINIATURE', 'MISERLY', 'MOD', 'MOM', 'MOONLIT', 'NAP', 'NAY', 'NIL', 'NITROUS', 'NOD', 'OBI', 'ODE', 'ONE', 'OPUS', 'OWL', 'OWN', 'PAR', 'PERFORM', 'PILOTED', 'PINEAPPLE', 'PITIFUL', 'PLEA', 'POI', 'POL', 'POLYMER', 'PONDEROUS', 'POW', 'PREEMPT', 'RAG', 'RAN', 'RED', 'REM', 'REP', 'RES', 'RESTORE', 'RETICENCE', 'RIMS', 'ROLL', 'ROUTINELY', 'RUN', 'RYE', 'SAY', 'SCENTED', 'SEAS', 'SEASIDE', 'SEE', 'SHUTS', 'SLEW', 'SONIC', 'SOY', 'SPA', 'SPEW', 'SPY', 'STEP', 'SUMO', 'SWOLLEN', 'TAM', 'TAN', 'TAR', 'TARIFFS', 'TEE', 'THRILLS', 'TIP', 'TIS', 'TOE', 'TOW', 'TOY', 'ULCER', 'UMP', 'USE', 'VIE', 'WALKING', 'WAN', 'WAR', 'WED', 'YEN']

def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 7) -> dict:
    word_source = word_source or _WORDS
    deadline = time.perf_counter() + 6.0
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    idx = _index_by_length(theme + fill)
    theme_set = set(theme)
    for _ in range(200):
        if time.perf_counter() > deadline:
            break
        white = _make_structure(size, rng)
        slots, _ = _slots_and_crossings(white, size)
        a = _fill(slots, idx, rng, theme_set, deadline=min(deadline, time.perf_counter() + 2.0))
        if a and len(a) == len(slots):
            return _build_layout(white, size, slots, a)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}