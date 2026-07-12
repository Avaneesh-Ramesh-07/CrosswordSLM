# === TASK CONTRACT (this program is written to satisfy the following) ===
# Task: From a natural language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) length >= 3
# letters; every white cell checked in BOTH directions; all white cells connected
# (one-track crossword); every entry a real word taken from word_source; high
# white-square density; completes within a few seconds.
# word_source is provided at runtime (a list, or a {"theme","fill"} dict of
# prioritized vocabulary + fill words); the curated word list is HARDCODED into _WORDS and used by default (word_source overrides it). Choose the
# layout algorithm (e.g. CSP backtracking with MRV + forward checking, AC-3
# maintenance, a (length,position,letter) pattern index, beam search, greedy
# longest-slot-first with a random component). Prefer packing long words where
# they will appear in 90-degree and 180-degree rotations, so a word placed at
# (1,1) can boost four entries. Consider also filling the main diagonal (a slot
# used by both across and down) and anti-diagonal. Scoring: words closest to
# longest-in-slot; highest white-square density; completes first.

"""gen2 fusion: csp_ac3 (AC-3 + MRV) + longest-first (randomized). AC-3's
reliable maintenance + MRV's focus on constrained slots, plus longest-first's
long-slot-packing bias. Unexplored combination - did AC-3 block long words
from long slots, or did longest-first miss a valid grid AC-3 could fill?

Self-contained; generate_crossword(topic, word_source=None, size). The curated vocabulary is HARDCODED into _WORDS and used by default; word_source is an optional override/fallback.
"""

import random
import time


def _split_source(word_source):
    if isinstance(word_source, dict):
        return word_source.get("theme", []), word_source.get("fill", [])
    return [], list(word_source)


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


def _slots_and_length(white, size):
    slots = [{"cells": cells, "len": length} for cells, length in _runs(white, size)]
    return slots


def _fill_slots(slots, idx, rng, theme_set, budget=8000, deadline=None):
    n = len(slots)
    dom = {si: set(idx.get(slots[si]["len"], [])) for si in range(n)}
    if any(not d for d in dom.values()):
        return None

    def revise(d, x, y):
        avail = {w[y] for w in d[x]}
        bd = {w for w in d[y] if w not in avail}
        if len(bd) != len(d[y]):
            d[x] &= avail
            return True
        return False

    def ac3(d, queue):
        while queue:
            if deadline is not None and time.perf_counter() > deadline:
                return False
            x, y = queue.pop()
            if revise(d, x, y):
                if not d[x]:
                    return False
                queue.extend((w, x) for w in d[x])
        return True

    if not ac3(dom, [(i, j) for i in range(n) for j in range(i + 1, n)]):
        return None

    used, assign, steps = set(), {}, [0]

    def bt(d):
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return None
        steps[0] += 1
        if not d:
            return assign.copy()
        si = min((s for s in d if len(d[s]) == 1), default=None)
        if not si:
            return None
        c = d[si].pop()
        assign[si] = c
        used.add(c)
        dead = any(w not in theme_set for w in c)
        if dead:
            d[si].add(c)
            assign.pop(si)
            return None
        result = bt(d)
        if result is not None:
            return result
        d[si].add(c)
        assign.pop(si)
        return None

    d = dict(dom)
    return bt(d) if bt(d) is not None and not assign else None


def _build_layout(white, size, slots, assignment):
    grid = {}
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            grid[(r, c)] = word[pos]
    numbers, num = {}, 0
    for si, word in assignment.items():
        for pos, (r, c) in enumerate(slots[si]["cells"]):
            numbers[(r, c)] = num
            num += 1
    cells = [dict(r=rc[0], c=rc[1], letter=grid[rc]) for rc in grid]
    across, down = [], []
    for cells, length in _runs(white, size):
        for i, (r, c) in enumerate(cells):
            num = numbers.get((r, c))
            if num is None:
                continue
            w = "".join(grid.get((r, cc), "") for cc in range(c, c + 1))
            if w.isalpha():
                across.append(dict(number=num, row=r, col=c, answer=w, length=length))
    for cells, length in _runs(white, size):
        for i, (r, c) in enumerate(cells):
            num = numbers.get((r, c))
            if num is None:
                continue
            w = "".join(grid.get((rr, c)) for rr in range(r, r + 1))
            if w.isalpha():
                down.append(dict(number=num, row=r, col=c, answer=w, length=length))
    return dict(rows=size, cols=size, cells=cells, across=across, down=down)


_WORDS = ['ACE', 'ACT', 'ADA', 'ADAM', 'ADS', 'ADM', 'ADO', 'AGILE', 'AGO', 'ALA', 'ALE', 'ALL', 'ALP', 'AMP', 'ANI', 'ANG', 'ANNA', 'APE', 'APT', 'ARE', 'ARK', 'ARR', 'ART', 'ASH', 'ASS', 'ASSOCIATE', 'ATE', 'AWAKEN', 'AXE', 'BAG', 'BAGGAGE', 'BAT', 'BED', 'BEETLES', 'BEN', 'BIN', 'BUREAU', 'CAP', 'CAR', 'CITE', 'CLOTHES', 'COALS', 'CODES', 'CODA', 'COLDEST', 'COLDEST', 'COO', 'COUP', 'CUSHION', 'CUR', 'CUSS', 'CUSHION', 'DAH', 'DAM', 'DAMAGES', 'DANCE', 'DAY', 'DEACON', 'DELIGHT', 'DEN', 'DESPERADO', 'DEVELOP', 'DEW', 'DIAPHRAGM', 'DINE', 'DIVE', 'DOC', 'DOE', 'DOH', 'DOH', 'DON', 'DOT', 'DRY', 'DUFFEL', 'DUN', 'EASEL', 'EAR', 'EARPHONES', 'EBB', 'EDEN', 'EDGE', 'EEL', 'EGO', 'EGOTISM', 'ELS', 'ELKS', 'ELS', 'ELM', 'ELM', 'EMIT', 'EMIR', 'EMU', 'EPITOME', 'ERA', 'ERR', 'ESP', 'ESP', 'ETA', 'EURO', 'EWE', 'EWE', 'FAD', 'FAT', 'FED', 'FEE', 'FIB', 'FIR', 'FLEA', 'FOB', 'FOG', 'FOG', 'GAG', 'GAGS', 'GAGS', 'GAL', 'GAS', 'GAS', 'GEM', 'GEM', 'GENERAL', 'GERMANE', 'GESTURE', 'GHOUL', 'HAT', 'HENS', 'HOPS', 'HOPS', 'HOPE', 'HUE', 'ICE', 'ICY', 'ILK', 'ILL', 'IMAGINE', 'IMP', 'INCLUSIVE', 'INE', 'ION', 'IRE', 'IRON', 'IRON', 'IRON', 'IRRESISTIBLE', 'ISLE', 'KNEE', 'LAP', 'LAP', 'LEA', 'LEA', 'LEG', 'LEI', 'LENT', 'LET', 'LIEU', 'LOG', 'LOT', 'LOT', 'LOVE', 'LOW', 'LYE', 'MAD', 'MACHETE', 'MACHETE', 'MACHETE', 'MAG', 'MAGGOT', 'MAGNET', 'MALARIA', 'MAN', 'MAS', 'MAS', 'MATES', 'MAX', 'MEAL', 'MEAN', 'MEET', 'MICROCOSM', 'MISERLY', 'MODELS', 'MODERATE', 'MONEY', 'MOOSE', 'MOUSE', 'MUM', 'NEE', 'NET', 'NEUTRAL', 'NEW', 'NUT', 'OAT', 'OAT', 'ODE', 'OIL', 'OLD', 'OMIT', 'ONE', 'ONE', 'OPUS', 'ORCA', 'ORCA', 'ORE', 'OUT', 'OVER', 'OWE', 'OWE', 'OWE', 'PAR', 'PAL', 'PAN', 'PANACHE', 'PANTHER', 'PANTHER', 'PAR', 'PART', 'PART', 'PARTNER', 'PATH', 'PAY', 'PEA', 'PEA', 'PEEP', 'PEOPLES', 'PESO', 'PET', 'PHONICS', 'PIC', 'PIC', 'PIP', 'PIRATE', 'PLEA', 'PONDEROUS', 'PORTRAY', 'POW', 'PRAIRIE', 'PRESIDE', 'PRO', 'PRO', 'PRO', 'PRO', 'RADIO', 'RAN', 'RAP', 'RAT', 'RED', 'REFLECT', 'REMODELED', 'RES', 'REST', 'REST', 'RESULT', 'RIM', 'RIP', 'ROAD', 'ROAR', 'ROB', 'RAT', 'RAT', 'REPRESENT', 'REPRISE', 'RES', 'RETIREE', 'RETREAT', 'REV', 'RIB', 'ROOK', 'ROOK', 'RUM', 'RUN', 'RUT', 'RUT', 'RYE', 'SALT', 'SAUCERS', 'SAY', 'SAY', 'SEA', 'SEAS', 'SEE', 'SEGMENT', 'SEI', 'SENSIBLE', 'SEP', 'SERUM', 'SEW', 'SEX', 'SIC', 'SKATE', 'SLEEP', 'SLY', 'SMIRK', 'SMIRK', 'SONIC', 'SOY', 'SPA', 'SPATIAL', 'SPECTRA', 'SPECTRE', 'SPICE', 'SPOILER', 'SPOONS', 'SPY', 'SPY', 'SPY', 'STAB', 'STEWARD', 'STEP', 'STIR', 'SUN', 'SUN', 'SUP', 'SURVEY', 'TACT', 'TACT', 'TACTS', 'TAN', 'TAR', 'TAX', 'TEA', 'TEE', 'TEENAGE', 'TELLING', 'TEN', 'TEN', 'TENT', 'TENTATIVE', 'THEREFORE', 'TIE', 'TIMOTHY', 'TIS', 'TOAD', 'TOAST', 'TONSILS', 'TOO', 'TOOL', 'TORE', 'TOTE', 'TOW', 'TOXIN', 'TOY', 'TOY', 'TREE', 'TREE', 'TRIP', 'TUNA', 'URN', 'URN', 'USE', 'VEST', 'VIETNAM', 'WAGE', 'WARHEAD', 'WEAPONS', 'WED', 'WET', 'WET', 'WIN', 'WINTER', 'WITH', 'WRY', 'YEAR', 'YETI', 'YETI', 'YON']

def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 7) -> dict:
    word_source = word_source or _WORDS
    deadline = time.perf_counter() + 6.0
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    idx = _index_by_length(theme + fill)
    for _ in range(200):
        if time.perf_counter() > deadline:
            break
        white = _make_structure(size, rng)
        slots = _slots_and_length(white, size)
        a = _fill_slots(slots, idx, rng, theme_set, budget=2000, deadline=min(deadline, time.perf_counter() + 2.0))
        if a and len(a) == len(slots):
            return _build_layout(white, size, slots, a)
    return dict(rows=size, cols=size, cells=[], across=[], down=[])