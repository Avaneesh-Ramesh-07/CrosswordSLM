# ==== INVALID unaugmented claude-opus-4-8 generation (clean-room, one-shot) ====
# Verdict from pipeline/eval_selfmodel.py scoring (English clean palette):
#   7x7: INVALID -> runner: timeout
#   9x9: INVALID -> runner: timeout
#   11x11: INVALID -> runner: timeout
# (word_source is injected at call time by the eval; this program uses it as its argument.)

import random
from collections import defaultdict


def generate_crossword(topic: str, word_source, size: int) -> dict:
    theme_words = [w.strip().upper() for w in word_source.get("theme", []) if w.strip()]
    fill_words = [w.strip().upper() for w in word_source.get("fill", []) if w.strip()]
    all_words = list(dict.fromkeys(theme_words + fill_words))

    # words by length
    by_len = defaultdict(list)
    for w in all_words:
        if w.isalpha():
            by_len[len(w)].append(w)

    # ---- generate a symmetric black-square pattern ----
    def make_pattern(n, seed):
        rnd = random.Random(seed)
        # start all white
        black = [[False] * n for _ in range(n)]
        # target black density
        target = int(n * n * (0.16 if n >= 11 else 0.10))
        placed = 0
        attempts = 0
        cells = [(r, c) for r in range(n) for c in range(n)]

        def valid_runs(bl):
            # every white run >= 3
            for r in range(n):
                run = 0
                for c in range(n):
                    if bl[r][c]:
                        if 0 < run < 3:
                            return False
                        run = 0
                    else:
                        run += 1
                if 0 < run < 3:
                    return False
            for c in range(n):
                run = 0
                for r in range(n):
                    if bl[r][c]:
                        if 0 < run < 3:
                            return False
                        run = 0
                    else:
                        run += 1
                if 0 < run < 3:
                    return False
            return True

        def connected(bl):
            start = None
            whites = 0
            for r in range(n):
                for c in range(n):
                    if not bl[r][c]:
                        whites += 1
                        if start is None:
                            start = (r, c)
            if start is None:
                return False
            seen = set([start])
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and not bl[nr][nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            return len(seen) == whites

        rnd.shuffle(cells)
        for (r, c) in cells:
            if placed >= target:
                break
            if r == n // 2 and c == n // 2 and n % 2 == 1:
                continue
            if black[r][c]:
                continue
            mr, mc = n - 1 - r, n - 1 - c
            black[r][c] = True
            black[mr][mc] = True
            if valid_runs(black) and connected(black):
                placed += 2 if (r, c) != (mr, mc) else 1
            else:
                black[r][c] = False
                black[mr][mc] = False
            attempts += 1

        if valid_runs(black) and connected(black):
            return black
        return None

    black = None
    for seed in range(400):
        p = make_pattern(size, seed)
        if p:
            black = p
            break
    if black is None:
        black = [[False] * size for _ in range(size)]

    n = size

    # ---- extract slots ----
    def get_slots(bl):
        across = []
        down = []
        for r in range(n):
            c = 0
            while c < n:
                if not bl[r][c]:
                    start = c
                    while c < n and not bl[r][c]:
                        c += 1
                    if c - start >= 3:
                        across.append(("A", r, start, c - start))
                else:
                    c += 1
        for c in range(n):
            r = 0
            while r < n:
                if not bl[r][c]:
                    start = r
                    while r < n and not bl[r][c]:
                        r += 1
                    if r - start >= 3:
                        down.append(("D", start, c, r - start))
                else:
                    r += 1
        return across + down

    slots = get_slots(black)

    # map cell -> slots covering it
    def slot_cells(slot):
        d, r, c, length = slot
        if d == "A":
            return [(r, c + i) for i in range(length)]
        else:
            return [(r + i, c) for i in range(length)]

    # ---- fill with backtracking ----
    grid = {}  # (r,c) -> letter

    # order slots to constrain intersections; longer first
    slots_sorted = sorted(slots, key=lambda s: -s[3])

    used = set()

    def pattern_for(slot):
        return "".join(grid.get(cell, ".") for cell in slot_cells(slot))

    def matches(pat, word):
        for pc, wc in zip(pat, word):
            if pc != "." and pc != wc:
                return False
        return True

    def solve(i):
        if i == len(slots_sorted):
            return True
        slot = slots_sorted[i]
        length = slot[3]
        pat = pattern_for(slot)
        candidates = by_len.get(length, [])
        # prefer theme words: shuffle but keep those first
        order = [w for w in candidates if matches(pat, w) and w not in used]
        random.shuffle(order)
        cells = slot_cells(slot)
        for word in order:
            changed = []
            ok = True
            for cell, ch in zip(cells, word):
                if cell in grid:
                    if grid[cell] != ch:
                        ok = False
                        break
                else:
                    grid[cell] = ch
                    changed.append(cell)
            if ok:
                used.add(word)
                if solve(i + 1):
                    return True
                used.discard(word)
            for cell in changed:
                del grid[cell]
        return False

    random.seed(12345)
    solved = solve(0)

    # ---- number the grid ----
    number = {}
    num = 1
    for r in range(n):
        for c in range(n):
            if black[r][c]:
                continue
            starts_across = (c == 0 or black[r][c - 1]) and (c + 1 < n and not black[r][c + 1])
            starts_down = (r == 0 or black[r - 1][c]) and (r + 1 < n and not black[r + 1][c])
            if starts_across or starts_down:
                number[(r, c)] = num
                num += 1

    cells = []
    for r in range(n):
        for c in range(n):
            if black[r][c]:
                cells.append({"r": r, "c": c, "letter": None})
            else:
                cell = {"r": r, "c": c, "letter": grid.get((r, c), "")}
                if (r, c) in number:
                    cell["number"] = number[(r, c)]
                cells.append(cell)

    across_out = []
    down_out = []
    for slot in slots:
        d, r, c, length = slot
        ans = "".join(grid.get(cc, "") for cc in slot_cells(slot))
        entry = {
            "number": number.get((r, c)),
            "row": r,
            "col": c,
            "answer": ans,
            "len": length,
        }
        if d == "A":
            across_out.append(entry)
        else:
            down_out.append(entry)

    across_out.sort(key=lambda e: (e["number"] or 0))
    down_out.sort(key=lambda e: (e["number"] or 0))

    return {
        "rows": n,
        "cols": n,
        "cells": cells,
        "across": across_out,
        "down": down_out,
    }
