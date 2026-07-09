# ==== VALID unaugmented claude-opus-4-8 generation (clean-room, one-shot) ====
# Verdict from pipeline/eval_selfmodel.py scoring (English clean palette):
#   7x7: INVALID -> runner: timeout
#   9x9: VALID  entries=30 cov=0.03 filler=0% rt=1.84s
#   11x11: INVALID -> runner: timeout
# (word_source is injected at call time by the eval; this program uses it as its argument.)

import random
from collections import defaultdict


def generate_crossword(topic: str, word_source, size: int) -> dict:
    theme = [w.strip().upper() for w in word_source.get("theme", []) if w.strip().isalpha()]
    fill = [w.strip().upper() for w in word_source.get("fill", []) if w.strip().isalpha()]
    all_words = list(dict.fromkeys(theme + fill))

    by_len = defaultdict(list)
    for w in all_words:
        by_len[len(w)].append(w)

    # Build word lookup by (length, position, char) for fast constraint checking
    def make_patterns(words):
        # index[(length)][(pos,char)] -> set of words
        idx = defaultdict(lambda: defaultdict(set))
        for w in words:
            L = len(w)
            for i, ch in enumerate(w):
                idx[L][(i, ch)].add(w)
        return idx

    idx = make_patterns(all_words)

    def matches(pattern):
        # pattern: string with None or char per position
        L = len(pattern)
        constraints = [(i, ch) for i, ch in enumerate(pattern) if ch is not None]
        if not constraints:
            return list(by_len.get(L, []))
        sets = [idx[L].get(c, set()) for c in constraints]
        if not sets or any(len(s) == 0 for s in sets):
            return []
        res = set(sets[0])
        for s in sets[1:]:
            res &= s
            if not res:
                return []
        return list(res)

    # --- Generate a symmetric black-square grid pattern ---
    def gen_pattern(n):
        # 0 = white, 1 = black
        best = None
        for attempt in range(300):
            grid = [[0] * n for _ in range(n)]
            # target black density
            max_black = int(n * n * (0.16 if n >= 11 else 0.12))
            placed = 0
            cells = [(r, c) for r in range(n) for c in range(n)]
            random.shuffle(cells)
            for (r, c) in cells:
                if placed >= max_black:
                    break
                rr, cc = n - 1 - r, n - 1 - c
                if grid[r][c] == 1:
                    continue
                grid[r][c] = 1
                grid[rr][cc] = 1
                if valid_runs(grid, n):
                    placed += 2 if (r, c) != (rr, cc) else 1
                else:
                    grid[r][c] = 0
                    grid[rr][cc] = 0
            if valid_runs(grid, n) and connected(grid, n):
                return grid
            if best is None:
                best = grid
        return [[0] * n for _ in range(n)]

    def valid_runs(grid, n):
        # every white run must be >= 3
        for r in range(n):
            run = 0
            for c in range(n + 1):
                if c < n and grid[r][c] == 0:
                    run += 1
                else:
                    if 0 < run < 3:
                        return False
                    run = 0
        for c in range(n):
            run = 0
            for r in range(n + 1):
                if r < n and grid[r][c] == 0:
                    run += 1
                else:
                    if 0 < run < 3:
                        return False
                    run = 0
        return True

    def connected(grid, n):
        whites = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]
        if not whites:
            return False
        seen = set()
        stack = [whites[0]]
        seen.add(whites[0])
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return len(seen) == len(whites)

    # --- Extract slots from a pattern ---
    def get_slots(grid, n):
        across = []
        down = []
        for r in range(n):
            c = 0
            while c < n:
                if grid[r][c] == 0:
                    start = c
                    while c < n and grid[r][c] == 0:
                        c += 1
                    if c - start >= 3:
                        across.append((r, start, c - start, 'A'))
                else:
                    c += 1
        for c in range(n):
            r = 0
            while r < n:
                if grid[r][c] == 0:
                    start = r
                    while r < n and grid[r][c] == 0:
                        r += 1
                    if r - start >= 3:
                        down.append((start, c, r - start, 'D'))
                else:
                    r += 1
        return across, down

    def cells_of(slot):
        r, c, L, d = slot
        if d == 'A':
            return [(r, c + i) for i in range(L)]
        return [(r + i, c) for i in range(L)]

    # --- Fill via backtracking ---
    def fill_grid(grid, n):
        across, down = get_slots(grid, n)
        slots = across + down
        # verify every white cell in both directions
        acr_cells = set()
        for s in across:
            acr_cells |= set(cells_of(s))
        dwn_cells = set()
        for s in down:
            dwn_cells |= set(cells_of(s))
        whites = {(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0}
        if acr_cells != whites or dwn_cells != whites:
            return None

        letters = {}  # (r,c) -> char
        assigned = {}  # slot -> word
        used = set()

        # order slots by number of options
        def slot_pattern(slot):
            return [letters.get(cell) for cell in cells_of(slot)]

        def solve(remaining):
            if not remaining:
                return True
            # pick slot with fewest candidates (MRV)
            best_slot = None
            best_cands = None
            for s in remaining:
                pat = slot_pattern(s)
                cands = [w for w in matches(pat) if w not in used]
                if best_cands is None or len(cands) < len(best_cands):
                    best_slot = s
                    best_cands = cands
                    if len(cands) == 0:
                        return False
            random.shuffle(best_cands)
            rest = [s for s in remaining if s != best_slot]
            cells = cells_of(best_slot)
            for w in best_cands:
                changed = []
                ok = True
                for cell, ch in zip(cells, w):
                    if cell in letters:
                        if letters[cell] != ch:
                            ok = False
                            break
                    else:
                        letters[cell] = ch
                        changed.append(cell)
                if ok:
                    used.add(w)
                    assigned[best_slot] = w
                    if solve(rest):
                        return True
                    del assigned[best_slot]
                    used.discard(w)
                for cell in changed:
                    del letters[cell]
            return False

        if solve(slots):
            return letters, across, down
        return None

    # --- Main loop: try patterns until fill succeeds ---
    result = None
    for attempt in range(200):
        grid = gen_pattern(size)
        if not valid_runs(grid, size) or not connected(grid, size):
            continue
        filled = fill_grid(grid, size)
        if filled:
            result = (grid, filled)
            break

    if result is None:
        # fallback empty grid
        grid = [[0] * size for _ in range(size)]
        letters = {}
        across, down = [], []
    else:
        grid, (letters, across, down) = result

    # --- Number the grid ---
    numbers = {}
    num = 1
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 1:
                continue
            starts_across = (c == 0 or grid[r][c - 1] == 1) and (c + 1 < size and grid[r][c + 1] == 0)
            starts_down = (r == 0 or grid[r - 1][c] == 1) and (r + 1 < size and grid[r + 1][c] == 0)
            if starts_across or starts_down:
                numbers[(r, c)] = num
                num += 1

    cells = []
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 1:
                continue
            cell = {"r": r, "c": c, "letter": letters.get((r, c), "")}
            if (r, c) in numbers:
                cell["number"] = numbers[(r, c)]
            cells.append(cell)

    across_out = []
    for (r, c, L, d) in across:
        word = "".join(letters.get((r, c + i), "") for i in range(L))
        across_out.append({"number": numbers.get((r, c)), "row": r, "col": c,
                           "answer": word, "len": L})
    down_out = []
    for (r, c, L, d) in down:
        word = "".join(letters.get((r + i, c), "") for i in range(L))
        down_out.append({"number": numbers.get((r, c)), "row": r, "col": c,
                         "answer": word, "len": L})

    across_out.sort(key=lambda x: (x["number"] or 0))
    down_out.sort(key=lambda x: (x["number"] or 0))

    return {
        "rows": size,
        "cols": size,
        "cells": cells,
        "across": across_out,
        "down": down_out,
    }
