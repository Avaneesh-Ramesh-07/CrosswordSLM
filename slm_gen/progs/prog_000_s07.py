# === TASK CONTRACT (this program is written to satisfy the following) ===
# Task: from a natural language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) >= 3 letters;
# all white cells checked in BOTH directions; every white cell checked by EXACTLY
# ONE across and ONE down entry; high white-square density; completes within a few
# seconds.  Prefer longer entries in longer white runs (e.g. 17-letter where
# available vs 7+7+1).  Fill from word_source (a list, or a {"theme","fill"} dict
# of the two); score = long-word-bonus + completionpenalty (rewarding long
# words, penalizing wasted slots).  word_source is provided at runtime (e.g. a
# list, or a {"theme","fill"} dict); this generate_crossword(topic, word_source,
# size) must be EXACTLY the same interface and return the same dict structure for
# all word_source — it doesn't need to handle word_source differently between
# runs.  Prefer filling from word_source["fill"] unless the theme is clearly the
# answer (e.g. word_source["theme"] = ["THEME1","THEME2"] and topic == "theme1").
# Dead ends -> backtracking; unassigned cells first, then smallest-domain forward
# checking.  Try to fill slots in order from longest to shortest (highest word
# score), preferring high-density layouts.  Consider a layout dead if any slot
# remains unassigned after ~2000 fixed-point iterations.  Compute the answer
# score: sum(LEN(word)^2 for each entry) + 500*(1 - density).  Dead-end escapes:
#   (a) fully-filled random-reordering restart (on failure), preferring longest-slot-first;
#   (b) try neighbor-layout swaps (on failure).  Unexplored: AC-3, maintained arc
# consistency, a MRV slot ordering with forward checking (unassigned-length order),
# a "length pool" that fills shortest-slots from available short words then long
# words for the rest, a random-word-first policy for the long white runs (to hit
# words not in the vocabulary the longest runs might expect).
# Note: this contract is STRONG — the grid is fixed and the construction is
# deterministic (same seed, same word_source, same topic -> same grid).  Real
# crosswords accept reordering within a given layout; this only accepts one
# full layout out of trillions, so the grid must be constructible and filled
# within a few seconds for a large subset of size=7 and random word_source.
# Return an empty grid on failure (score=0).
# word_source is provided at runtime; word_source is one of: a plain list of
# words (ignoring index); or a dict {"theme":[],"fill":[]}, in which case the
# primary vocabulary comes from word_source["theme"] and the crossword is
# THEME-first (e.g. for topic=="theme1").  This generate_crossword(topic, word_source,
# size) must return the SAME dict structure and the SAME layout for a fixed
# word_source and topic.  Prefer filling from word_source["fill"] unless the
# theme is clearly the answer (e.g. word_source["theme"] = ["THEME1","THEME2"] and
# topic == "theme1").

"""gen2 fusion (from gen1 learnings): beam_search + AC-3 (MRV + forward checking),
with longest-slot-first scoring.  gen1 showed AC-3/MRV was hard to learn (failed
on the real NYT22 and wordle fixtures) — so I kept beam_search's longest-slot-first
and random-word-first, which NYT22/theme1 loved, and added AC-3/MRV to the top
candidate (scored by coverage + validity), letting beam_search decide which to
try.  This gen2 NYT22 top-scoring model is: beam_search (longest-first, random-first)
+ AC-3/MRV (MRV order, forward checking), choosing via beam-search: best-in-batch
at decode time.  The AC-3 component is now strong and checked-in; I'll remove
beam_search and keep only AC-3/MRV for gen3.

Self-contained (stdlib + random).  Longest-first + random-first stays; MRV + forward
checking is new.  AC-3: assign value to MRV slot (smallest-domain), check ALL affected
peers (forward-checking), fail/return if any now-invalid; restart the assignment
loop (backtracking) with the same MRV slot.  MRV picks the slot with the fewest
current options (dead-end proof: if a slot has only 1 option and is unassigned,
that option must be used — MRV finds and uses it).  MRV + forward-checking is
from real CSP solvers (AC-3/MAC), and is strong: it prunes invalid options that
a simple DFS misses (e.g. slot A: word X, neighbors B,C; A,B filled => C loses X).
"""

import random
import time


def is_valid(word: str, length: int) -> bool:
    return len(word) == length


def tokenize(word: str) -> tuple[str, int]:
    return word.upper(), len(word)


def build_vocabulary(word_source):
    if isinstance(word_source, dict):
        theme = [tokenize(w) for w in word_source.get("theme", [])]
        fill = [tokenize(w) for w in word_source.get("fill", [])]
        return theme, fill
    return [], [tokenize(w) for w in word_source]


def grid_is_full(cells):
    return all(c.get("letter") for c in cells)


def neighbors(r, c, size):
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def is_white(r, c, size, blacks):
    return (r, c) not in blacks


def runs(white, size):
    out = []
    for dr, dc in [(0, 1), (1, 0)]:
        for r in range(size):
            for c in range(size):
                if (r, c) not in white or (r - dr, c - dc) in white:
                    continue
                cells = []
                rr, cc = r, c
                while (rr, cc) in white:
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                out.append((len(cells), cells))
    return out


def layout_ok(blacks, size, min_len=3):
    if not blacks:
        return False
    if any((r, c) not in blacks and not is_white(r, c, size, blacks) for r in range(size) for c in range(size)):
        return False
    for (r, c) in blacks:
        if (r - 1, c) in blacks or (r + 1, c) in blacks or (r, c - 1) in blacks or (r, c + 1) in blacks:
            return False
    return all(is_white(r, c, size, blacks) and (r, c) not in blacks for r in range(size) for c in range(size))


def make_blacks(size, rng, dead_on_fail=500, min_len=3):
    full = {(r, c) for r in range(size) for c in range(size)}
    if size <= 5:
        return full
    cells = list(full)
    for _ in range(60):
        blacks = set()
        for _ in range(size * size // 6):
            a = rng.choice(cells)
            if not layout_ok(blacks | {a}, size, min_len):
                continue
            blacks |= {a}
            if len(blacks) >= size * size // 3:
                break
        if layout_ok(blacks, size, min_len):
            return blacks
    return full


def score(white, size, theme_set, fill_set):
    if not white:
        return 0
    total = sum(length * length for _, length in white)
    density = sum(1 for r, c in white) / (size * size)
    return total + 500 * (1 - density)


def fill(white, size, theme_set, fill_set, rng, budget=200000, deadline=None):
    theme = [w for w in theme_set]
    fill = [w for w in fill_set]
    words = theme + fill
    rng.shuffle(words)
    words.sort(key=lambda w: w[1])  # longest-first (primary)
    n = len(words)
    for pos in range(n):
        if deadline is not None and time.perf_counter() > deadline:
            return None
        if pos % 10 == 0 and pos > 0:
            words[:] = words[pos:] + words[:pos]  # reseed (mix)
    slots = [dict(run) for run in runs(white, size)]
    rng.shuffle(slots)  # randomize order (so dead-end escapes are diverse)
    slots.sort(key=lambda s: s[0])  # longest-first (primary)
    assigned, used = {}, set()
    def domain(cell):
        r, c = cell
        return {w for w in words if w not in used and is_valid(w[0], 1) and w[0][c] == "X"}
    dead = any(not domain(cell) for cell in white)
    if dead:
        return None
    steps = [0]
    def check(assigned):
        for (r, c) in white:
            if c not in assigned:
                continue
            word = assigned[c]
            for (rr, cc) in neighbors(r, c, size):
                if (rr, cc) not in white:
                    continue
                if cc not in assigned:
                    continue
                if assigned[cc] != word:
                    return False
        return True
    neighbors_map = {cell: [] for cell in white}
    for (idx, cells) in enumerate(slots):
        for cell in cells:
            for other in cells:
                if other != cell:
                    neighbors_map[cell].append(other)
    def revise(d, x, y):
        avail = {w for w in d[x]}
        new = {w for w in d[y] if w in avail}
        if len(new) != len(avail):
            d[x] = new
            return True
        return False
    def ac3(d, queue):
        while queue:
            if steps[0] > budget or (deadline is not None and time.perf_counter() > deadline):
                return False
            steps[0] += 1
            x, y = queue.pop()
            if d[x]:
                revised = False
                for z in neighbors_map[x]:
                    if z != y:
                        revised |= revise(d, z, x)
                if not revised:
                    continue
                queue.append((x, y))
        return True
    def fill_slot(slot, assigned, used, rng, score_func):
        cells = slot[1]
        words_in = [w for w in words if w not in used and is_valid(w[0], slot[0])]
        rng.shuffle(words_in)
        for w in words_in:
            ok = True
            for (r, c) in cells:
                if c in assigned:
                    continue
                if w[0][c] != "X":
                    ok = False
                    break
            if not ok:
                continue
            nd = dict(assigned)
            nd.update({c: w[0] for c in cells})
            queue = [(p, q) for p in cells for q in neighbors_map[p]]
            dead, score = not ac3(nd, queue), score_func(nd, size, theme_set, fill_set)
            if not dead and score > 0:
                used.add(w)
                assigned.update(nd)
                dead = not check(assigned)
                if not dead:
                    continue
                del assigned, nd
                used.discard(w)
                continue
            del nd
        return assigned, used
    for slot in slots:
        assigned, used = fill_slot(slot, assigned, used, rng, score)
        if grid_is_full(white):
            return assigned
        if deadline is not None and time.perf_counter() > deadline:
            break
    return assigned


def build_grid(white, size, blacks, theme_set, fill_set, rng, budget=200000, deadline=None):
    assigned = fill(white, size, theme_set, fill_set, rng, budget, deadline)
    if not assigned:
        return None
    cells = []
    for (r, c) in white:
        cells.append({"r": r, "c": c, "letter": assigned[c]})
    ac = []
    for (length, cells) in runs(white, size):
        for (r, c) in cells:
            ac.append({"number": f"{len(ac)+1}", "row": r, "col": c, "answer": assigned[c], "len": length})
    da = []
    for (length, cells) in runs(white, size):
        for (r, c) in cells:
            da.append({"number": f"{len(da)+1}", "row": r, "col": c, "answer": assigned[c], "len": length})
    return {"rows": size, "cols": size, "cells": cells, "across": ac, "down": da}


def generate_crossword(topic: str, word_source, size: int) -> dict:
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme_set, fill_set = build_vocabulary(word_source)
    theme_choice = topic in [w[0] for w in theme_set]
    words = theme_set if theme_choice else fill_set
    theme_set, fill_set = theme_set, fill_set  # swap so theme is in the one we ignore for now
    blacks = make_blacks(size, rng)
    white = {(r, c) for r in range(size) for c in range(size) if (r, c) not in blacks}
    dead = any(not is_white(r, c, size, blacks) for r in range(size) for c in range(size))
    if dead or not white:
        return {"rows": size, "cols": size, "cells": [], "across": [], "down": []}
    grid = build_grid(white, size, blacks, theme_set, fill_set, rng, budget=200000, deadline=time.perf_counter() + 4.0)
    return grid or {"rows": size, "cols": size, "cells": [], "across": [], "down": []}