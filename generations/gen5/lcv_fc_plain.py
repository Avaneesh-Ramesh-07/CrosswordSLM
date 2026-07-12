"""gen5 fusion: MRV + pure least-constraining-value + forward checking, template-scaled.

Parents: reference_v1 (helpers/templates), gen4 mrv_fc_pidx (palette-scaling fix).
Technique: minimum-remaining-values slot order with a PURE least-constraining-value
value order (no theme priority, no frequency term) and forward checking. Occupies
the LCV/plain/forward-checking design cell. Self-contained, stdlib only; every
answer comes from word_source. Standard layout schema."""

import random
import time
from collections import deque


def _split_source(word_source):
    """Accept a plain list OR a {"theme","fill"} dict. -> (theme, fill) upper lists."""
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
                if (r, c) not in white:
                    continue
                if (r - dr, c - dc) in white:
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
    """Verbatim reference_v1 structure builder (used for size 7)."""
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
            trial_white = full - (blacks | {(r, c), partner})
            if _structure_ok(trial_white, size, min_len):
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


def _build_pattern_index(idx_by_len):
    """(length, position, letter) -> set(words). Built ONCE per call and reused
    across every structure attempt (the palette-scaling fix)."""
    pat = {}
    for length, words in idx_by_len.items():
        for w in words:
            for pos, ch in enumerate(w):
                pat.setdefault((length, pos, ch), set()).add(w)
    return pat


def _crossings(slots):
    """For each slot: list of (my_pos, other_slot, other_pos) at each shared cell,
    plus a parallel list of slot lengths. In an all-checked grid every white cell
    is in exactly one across + one down slot, so each shared cell yields one arc."""
    cellpos = {}
    for si, s in enumerate(slots):
        for pos, cell in enumerate(s["cells"]):
            cellpos.setdefault(cell, []).append((si, pos))
    cross = [[] for _ in range(len(slots))]
    for lst in cellpos.values():
        if len(lst) == 2:
            (a, pa), (b, pb) = lst
            cross[a].append((pa, b, pb))
            cross[b].append((pb, a, pa))
    return cross, [s["len"] for s in slots]


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


_TEMPLATES_7 = [
    [[0, 3], [1, 3], [3, 0], [3, 1], [3, 5], [3, 6], [5, 3], [6, 3]],
    [[0, 3], [3, 0], [3, 1], [3, 5], [3, 6], [6, 3]],
    [[0, 3], [1, 3], [3, 0], [3, 6], [5, 3], [6, 3]],
    [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [2, 0], [3, 0], [3, 6], [4, 6], [5, 5], [5, 6], [6, 3], [6, 4], [6, 5], [6, 6]],
]

# Curated size-9 subset: gen4 templates that fill 8/8 within ~0.95s on the
# purified palette (the full gen4 list is mostly slow, unsafe to shuffle).
_TEMPLATES_9 = [
    [[0, 4], [1, 4], [2, 4], [3, 0], [3, 5], [4, 0], [4, 1], [4, 7], [4, 8], [5, 3], [5, 8], [6, 4], [7, 4], [8, 4]],
    [[0, 4], [1, 4], [2, 4], [3, 3], [3, 8], [4, 0], [4, 1], [4, 7], [4, 8], [5, 0], [5, 5], [6, 4], [7, 4], [8, 4]],
    [[0, 4], [1, 4], [3, 3], [3, 8], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 0], [5, 5], [7, 4], [8, 4]],
    [[0, 4], [3, 3], [3, 7], [3, 8], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 0], [5, 1], [5, 5], [8, 4]],
    [[0, 3], [0, 8], [3, 0], [3, 5], [4, 0], [4, 1], [4, 2], [4, 6], [4, 7], [4, 8], [5, 3], [5, 8], [8, 0], [8, 5]],
    [[0, 4], [0, 8], [1, 4], [2, 4], [3, 0], [3, 1], [3, 5], [5, 3], [5, 7], [5, 8], [6, 4], [7, 4], [8, 0], [8, 4]],
]

# Size-11 fillable subset: on the purified palette only a few gen4 structures
# fill at all (most are ~0% fillable within the per-attempt budget). Restricting
# the pool to these -- best-first, then shuffled+cycled by _structures -- lets the
# 12s budget cycle them enough for parent-level reliability (the dead templates
# would otherwise waste attempts and tank a shuffled draw).
_TEMPLATES_11 = [
    [[0, 0], [0, 4], [1, 4], [2, 4], [3, 5], [3, 6], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [7, 4], [7, 5], [8, 6], [9, 6], [10, 6], [10, 10]],
    [[0, 4], [0, 5], [0, 10], [1, 4], [2, 4], [3, 3], [3, 7], [4, 6], [5, 0], [5, 1], [5, 9], [5, 10], [6, 4], [7, 3], [7, 7], [8, 6], [9, 6], [10, 0], [10, 5], [10, 6]],
    [[0, 4], [0, 10], [1, 4], [2, 4], [3, 6], [4, 3], [4, 8], [4, 9], [4, 10], [5, 3], [5, 7], [6, 0], [6, 1], [6, 2], [6, 7], [7, 4], [8, 6], [9, 6], [10, 0], [10, 6]],
    [[0, 3], [0, 7], [1, 3], [1, 7], [3, 0], [3, 5], [3, 9], [3, 10], [4, 6], [5, 3], [5, 7], [6, 4], [7, 0], [7, 1], [7, 5], [7, 10], [9, 3], [9, 7], [10, 3], [10, 7]],
]


def _structures(size, rng):
    """Yield candidate white-cell sets. Sizes 7/9/11 draw from pre-verified
    fillable templates (shuffled per call for output diversity, then cycled);
    any other size falls back to random symmetric construction."""
    full = {(r, c) for r in range(size) for c in range(size)}
    tpls = {7: _TEMPLATES_7, 9: _TEMPLATES_9, 11: _TEMPLATES_11}.get(size)
    if not tpls:
        while True:
            yield _make_structure(size, rng)
    else:
        idx = list(range(len(tpls)))
        rng.shuffle(idx)
        while True:
            for i in idx:
                yield full - {(r, c) for (r, c) in tpls[i]}


_CAP = 2000
_LCV_WINDOW = 20   # head window that is fully LCV-scored (keeps ordering cheap)
_LCV_SCAN = 300    # cap the neighbour-domain scan when counting LCV support


def _solve(slots, cross, slen, by_len, pat, LF, rng, deadline, theme_set):
    """Minimum-remaining-values + PURE least-constraining-value + forward checking.

    Slot choice is minimum-remaining-values. Value choice is pure LCV: within the
    head window of the candidate list, words are ranked by how many options they
    LEAVE OPEN across unassigned neighbours (counted against the live domains), and
    the most-open word is tried first -- no theme priority and no frequency term.
    Forward checking prunes neighbour domains after each placement. This occupies
    the LCV-ordered / plain / forward-checking cell of the design space, distinct
    from mac_lcv_theme (MAC core, theme-first) and from the frequency-ordered
    dsatur / domwdeg fusions."""
    n = len(slots)
    dom = []
    for s in slots:
        p = list(by_len.get(s["len"], []))
        rng.shuffle(p)
        dom.append(p[:_CAP])
    assign, used = {}, set()

    def bt():
        if time.perf_counter() > deadline:
            return None
        if len(assign) == n:
            return True
        best, bs = -1, 1 << 30
        for si in range(n):
            if si in assign:
                continue
            l = len(dom[si])
            if l < bs:
                bs, best = l, si
        if bs == 0:
            return False
        si = best
        crs = cross[si]
        cands = [w for w in dom[si] if w not in used]

        def lcv(w):
            t = 1
            for (mp, nsi, npos) in crs:
                if nsi in assign:
                    continue
                ch = w[mp]
                t += sum(1 for x in dom[nsi][:_LCV_SCAN] if x[npos] == ch)
            return t

        head = cands[:_LCV_WINDOW]
        head.sort(key=lambda w: -lcv(w))
        cands = head + cands[_LCV_WINDOW:]
        for w in cands:
            if time.perf_counter() > deadline:
                return None
            assign[si] = w
            used.add(w)
            rem, ok = [], True
            for (mp, nsi, npos) in crs:
                if nsi in assign:
                    if assign[nsi][npos] != w[mp]:
                        ok = False
                        break
                    continue
                ch = w[mp]
                d = dom[nsi]
                keep = [x for x in d if x[npos] == ch and x != w]
                rem.append((nsi, d))
                dom[nsi] = keep
                if not keep:
                    ok = False
                    break
            r = bt() if ok else False
            if r is True:
                return True
            for (nsi, d) in rem:
                dom[nsi] = d
            del assign[si]
            used.discard(w)
            if r is None:
                return None
        return False

    return dict(assign) if bt() is True else None


def generate_crossword(topic, word_source, size):
    budgets = {5: 2.0, 7: 3.0, 9: 5.0, 11: 12.0, 13: 20.0, 15: 30.0}
    per_attempt = {5: 0.6, 7: 0.7, 9: 1.1, 11: 1.4}.get(size, 1.2)
    deadline = time.perf_counter() + budgets.get(size, 5.0) - 0.4
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    by_len = _index_by_length(theme + fill)
    empty = {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
    if not by_len:
        return empty
    # PALETTE-SCALING FIX: build the (length,pos,letter) pattern index ONCE and
    # reuse it across every structure attempt; LF = per-bucket size for ordering.
    pat = _build_pattern_index(by_len)
    LF = {k: len(v) for k, v in pat.items()}
    for white in _structures(size, rng):
        if time.perf_counter() > deadline:
            break
        slots, _ = _slots_and_crossings(white, size)
        cross, slen = _crossings(slots)
        sub = min(deadline, time.perf_counter() + per_attempt)
        assignment = _solve(slots, cross, slen, by_len, pat, LF, rng, sub, theme_set)
        if assignment and len(assignment) == len(slots):
            return _build_layout(white, size, slots, assignment)
    return empty


if __name__ == "__main__":
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    import pipeline.regen_sft as R
    pal = R.load_purified()
    for sz in (7, 9, 11):
        t0 = time.perf_counter()
        lay = generate_crossword("vocabulary", pal["ws"], sz)
        print("size %d: %dA %dD, %d white cells  (%.1fs)" % (
            sz, len(lay["across"]), len(lay["down"]), len(lay["cells"]),
            time.perf_counter() - t0))
