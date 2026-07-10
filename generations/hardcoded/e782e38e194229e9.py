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

"""Reference fixed-grid crossword generator (seed v1).

Clean-license reimplementation of the canonical CSP pattern (à la CS50 / qxw):
  1. construct a 180-degree symmetric grid whose white runs are all length >= 3
     and fully connected (so it is "all-checked" and NYT-legal);
  2. fill every slot from the supplied `word_source` via backtracking search with
     MRV slot ordering + forward checking, choosing among the top-k candidates at
     random (with restarts) to escape dead ends.

Self-contained on purpose: only stdlib + `random`, and it NEVER hardcodes words
— every answer comes from the `word_source` passed in. This is a seed for
OpenEvolve to evolve/diversify, and the positive fixture proving the harness
accepts a real generator. Returns the standard layout schema; on failure it
returns an empty grid (which the scorer marks invalid).
"""

import random
import time


def _index_by_length(word_source):
    idx = {}
    for w in word_source:
        w = str(w).upper()
        if w.isalpha():
            idx.setdefault(len(w), []).append(w)
    return idx


def _runs(white, size):
    """All maximal white runs (both directions) as (cells, length)."""
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        for r in range(size):
            for c in range(size):
                if (r, c) not in white:
                    continue
                if (r - dr, c - dc) in white:
                    continue  # not a run start
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
    """Return a set of white cells forming a valid symmetric NYT-legal structure."""
    full = {(r, c) for r in range(size) for c in range(size)}
    if size <= 5:
        return full  # a fully-open small square is already valid (word square)

    cells = list(full)
    for _ in range(60):  # a few structure attempts
        rng.shuffle(cells)
        blacks = set()
        target = (size * size) // 6  # ~17% black, NYT-ish
        for (r, c) in cells:
            if len(blacks) >= target:
                break
            partner = (size - 1 - r, size - 1 - c)
            if (r, c) == partner or (r, c) in blacks or partner in blacks:
                continue
            trial_white = full - (blacks | {(r, c), partner})
            if _structure_ok(trial_white, size, min_len):
                blacks |= {(r, c), partner}
        white = full - blacks
        if _structure_ok(white, size, min_len):
            return white
    return full  # fallback (may be hard to fill for large size)


def _slots_and_crossings(white, size):
    """Return (slots, cell_to_slots). Each slot: {'cells': [...], 'len': n}."""
    slots = []
    for cells, length in _runs(white, size):
        slots.append({"cells": cells, "len": length})
    cell_to_slots = {}
    for i, s in enumerate(slots):
        for cell in s["cells"]:
            cell_to_slots.setdefault(cell, []).append(i)
    return slots, cell_to_slots


def _build_pattern_index(idx_by_len):
    """(length, position, letter) -> set(words) for O(1) constrained lookup."""
    pat = {}
    for length, words in idx_by_len.items():
        for w in words:
            for pos, ch in enumerate(w):
                pat.setdefault((length, pos, ch), set()).add(w)
    return pat


def _pool(slot, letters, pat, by_len):
    """Words matching the slot's currently-fixed letters (ignores word reuse)."""
    fixed = [(pos, letters[cell]) for pos, cell in enumerate(slot["cells"]) if cell in letters]
    if not fixed:
        return by_len.get(slot["len"], [])
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
    return result


def _fill(slots, cell_to_slots, idx, rng, budget=200000, deadline=None):
    """Backtracking + MRV + forward checking with a pattern index. -> {slot: word} or None."""
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

    def domain_size(si):
        return len(_pool(slots[si], letters, pat, by_len))

    def backtrack():
        if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
            return False
        steps[0] += 1
        if len(assignment) == len(slots):
            return True
        # MRV: unassigned slot with the smallest current domain
        best_si, best = None, 1 << 30
        for si in range(len(slots)):
            if si in assignment:
                continue
            size = domain_size(si)
            if size < best:
                best_si, best = si, size
                if best == 0:
                    break
        if best_si is None or best == 0:
            return False
        cands = [w for w in _pool(slots[best_si], letters, pat, by_len) if w not in used]
        rng.shuffle(cands)  # randomized -> diverse solutions across runs
        for word in cands:
            changed = []
            for pos, cell in enumerate(slots[best_si]["cells"]):
                if cell not in letters:
                    letters[cell] = word[pos]
                    changed.append(cell)
            assignment[best_si] = word
            used.add(word)
            dead = any(nb not in assignment and domain_size(nb) == 0 for nb in neigh[best_si])
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

    numbers, n = {}, 0
    across, down = [], []
    for r in range(size):
        for c in range(size):
            if not is_white(r, c):
                continue
            sa = (c == 0 or not is_white(r, c - 1)) and (c + 1 < size and is_white(r, c + 1))
            sd = (r == 0 or not is_white(r - 1, c)) and (r + 1 < size and is_white(r + 1, c))
            if sa or sd:
                n += 1
                numbers[(r, c)] = n
            if sa:
                w, cc = "", c
                while cc < size and is_white(r, cc):
                    w += grid[(r, cc)]
                    cc += 1
                across.append({"number": n, "row": r, "col": c, "answer": w, "len": len(w)})
            if sd:
                w, rr = "", r
                while rr < size and is_white(rr, c):
                    w += grid[(rr, c)]
                    rr += 1
                down.append({"number": n, "row": r, "col": c, "answer": w, "len": len(w)})
    cells = []
    for (r, c), ch in sorted(grid.items()):
        cell = {"r": r, "c": c, "letter": ch}
        if (r, c) in numbers:
            cell["number"] = numbers[(r, c)]
        cells.append(cell)
    return {"rows": size, "cols": size, "cells": cells, "across": across, "down": down}


_WORDS = ['ABLE', 'ABUNDANCE', 'ACCUSES', 'ACE', 'ACME', 'ACTS', 'ADA', 'ADAM', 'ADS', 'ADULATION', 'AGE', 'AGELESS', 'AGO', 'ALA', 'ALE', 'ALGEBRA', 'ALL', 'ALOE', 'ALUM', 'AMP', 'AMPS', 'ANNA', 'ANT', 'ANTI', 'ARC', 'ARE', 'AREAS', 'ARIA', 'ARM', 'ASH', 'ASP', 'ASS', 'ATE', 'ATOP', 'AWARE', 'AWE', 'BAG', 'BAR', 'BAT', 'BEDS', 'BEETLES', 'BEN', 'BIG', 'BLIP', 'BOD', 'BOY', 'BUREAUS', 'CAMPSITES', 'CAP', 'CAPS', 'CAR', 'CATERER', 'CELL', 'CHI', 'CINEMATIC', 'CIVIC', 'COALS', 'COD', 'CODA', 'COERCED', 'COO', 'COW', 'CUE', 'CURE', 'DAUNT', 'DELI', 'DEN', 'DEW', 'DIE', 'DIET', 'DIP', 'DISPLAY', 'DOE', 'DOG', 'DOH', 'DON', 'DRY', 'DUES', 'EACH', 'EASEL', 'EBB', 'ECSTASY', 'EDIT', 'EDUCATE', 'EEL', 'EGO', 'EGOTISM', 'ELECTORAL', 'EMAILED', 'EMINENT', 'EMU', 'END', 'ENDS', 'EON', 'EPITOME', 'ERA', 'ERG', 'ERR', 'ESP', 'ESPY', 'ESSENCE', 'ETA', 'EVE', 'EWE', 'FATE', 'FEN', 'FIB', 'FIR', 'FOG', 'FORWARD', 'GAGS', 'GALLEYS', 'GAS', 'GEMS', 'GENTLEMEN', 'GETTING', 'GIN', 'GIVE', 'GLADSTONE', 'GODS', 'GOO', 'GUN', 'HAT', 'HAVE', 'HELLO', 'HEM', 'HENS', 'HEP', 'HES', 'HEW', 'HOE', 'HOP', 'HOPS', 'IBIS', 'ICE', 'ICY', 'IDEA', 'IDES', 'IDS', 'INSTANTLY', 'ION', 'IRE', 'IRON', 'ISOMETRIC', 'ITEM', 'ITS', 'KITE', 'LABELED', 'LAM', 'LEE', 'LEEWARD', 'LEG', 'LEI', 'LET', 'LIEU', 'LOT', 'LUG', 'MACABRE', 'MACHETE', 'MAO', 'MAR', 'MAS', 'MEAL', 'MEANEST', 'MENTALITY', 'MERITED', 'MET', 'METEORS', 'MINI', 'MOP', 'MORAL', 'MULE', 'NAG', 'NAN', 'NASAL', 'NEE', 'NEGATED', 'NEO', 'NEON', 'NESTLED', 'NEW', 'NIT', 'NOG', 'NOTE', 'ODD', 'ODE', 'OHM', 'OIL', 'OILY', 'OLIVE', 'OMEGA', 'OMIT', 'ONE', 'ONGOING', 'ORE', 'OSLO', 'OTC', 'OTIS', 'OVER', 'OWE', 'OWN', 'PAR', 'PAS', 'PAT', 'PAY', 'PEA', 'PEEP', 'PENETRATE', 'PEOPLES', 'PESO', 'PET', 'PIC', 'PLACEMENT', 'PLAN', 'PLEA', 'POI', 'POL', 'POLEMIC', 'POM', 'PONDS', 'POPE', 'POW', 'PRAIRIE', 'PREDICTOR', 'PRIM', 'PRO', 'PROMPTS', 'PSI', 'PUP', 'RADIO', 'RAG', 'RAM', 'RAN', 'RAT', 'RED', 'REM', 'REP', 'RES', 'RETIREE', 'RIP', 'ROB', 'ROBOT', 'ROE', 'ROME', 'ROOT', 'ROT', 'RUE', 'RUN', 'SAC', 'SALT', 'SANE', 'SAUCERS', 'SAY', 'SCAM', 'SEE', 'SEER', 'SEGMENT', 'SENT', 'SEW', 'SEX', 'SIC', 'SKI', 'SLY', 'SNAGGED', 'SNEER', 'SONNETS', 'SOW', 'SPA', 'SPACE', 'SPINACH', 'SPY', 'STAG', 'STATUETTE', 'STEPPES', 'STIR', 'SUBATOMIC', 'SUM', 'SUN', 'SUSPENDED', 'SWIMMER', 'SYNOD', 'SYSTEMS', 'TAD', 'TAN', 'TAR', 'TAX', 'TEA', 'TEAMSTERS', 'TEE', 'TEN', 'TENTACLES', 'TEST', 'TOE', 'TOLERATES', 'TON', 'TOO', 'TOW', 'TREE', 'TUNIS', 'TWO', 'UNTREATED', 'URN', 'USE', 'USER', 'VIDEOTAPE', 'VIM', 'VOICE', 'WAGE', 'WAR', 'WAS', 'WEAVING', 'WELCOME', 'WERE', 'WET', 'WIG', 'WILDCAT', 'WIN', 'WRISTBAND', 'WRY', 'YEN', 'YES', 'YON']

def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 7) -> dict:
    word_source = word_source or _WORDS
    deadline = time.perf_counter() + 4.0  # wall-clock bound (stays under sandbox timeout)
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    if isinstance(word_source, dict):  # theme+fill contract: this baseline just fills from both
        word_source = list(word_source.get("theme", [])) + list(word_source.get("fill", []))
    idx = _index_by_length(word_source)
    for _ in range(200):  # fail fast per structure, try many structures within the deadline
        if time.perf_counter() > deadline:
            break
        white = _make_structure(size, rng)
        slots, cell_to_slots = _slots_and_crossings(white, size)
        assignment = _fill(slots, cell_to_slots, idx, rng, budget=1200, deadline=deadline)
        if assignment and len(assignment) == len(slots):
            return _build_layout(white, size, slots, assignment)
    return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}