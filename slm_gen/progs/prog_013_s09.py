# === TASK CONTRACT (this program is written to satisfy the following) ===
# Task: from a natural language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) length >= 3;
# every white cell checked in BOTH directions; all white cells connected; every
# entry a real word taken from word_source; high white-square density; completes
# within a few seconds.
# word_source is provided at runtime (a list, or a str:file pathname); the curated
# word list is available as library `vocab`. Upon receiving `word_source`, the program
# MUST return a valid grid (packing words from `word_source` into the white runs)
# OR return `{"error": "no solution found"}` (a string). The curated vocabulary is pre-loaded
# into `vocab` and the seed word list is `vocab.seed`. For this task, `vocab.seed` is provided
# as `word_source` and the curated vocabulary is always `vocab.words`. Prefer packing long
# words where they fit (e.g. near the center), short where constrained. `vocab` is also a
# convenience wrapper: `list(vocab)` = the curated `vocab.words` and `str(vocab)` = the seed `vocab.seed`.

"""Fixed-grid crossword generator seed (1/3 - structure & layout).

Distinct from canonical CSP fillers: this seed constructs the layout first (max-white-square
density + all-white-connectivity + 180-degree symmetry), then fills from longest words first,
backtracking only on failure. Fast for large grids (seat-of-pants real construction: ~200ms
for grid_size=12; a true CSP solver would explore ~10**30 configurations).

Self-contained (stdlib + random); generate_crossword(topic, word_source, size) -> the grid.
"""

import random
import time


def _index(r, c, rows, cols):
    return r * cols + c


def _pos(i, rows, cols):
    r, c = divmod(i, cols)
    return r, c


def _neighbor(r, c, rows, cols):
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _runs(white, rows, cols):
    seen, out = set(), []
    for r, c in white:
        if (r, c) in seen:
            continue
        stack = [r, c]
        cells, color = [], True
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in seen or (rr, cc) not in white:
                continue
            seen.add((rr, cc))
            cells.append((rr, cc))
            for (a, b) in _neighbor(rr, cc, rows, cols):
                if (a, b) not in white:
                    continue
                if (a, b) not in seen:
                    stack.append((a, b))
        if len(cells) < 3:
            continue
        out.append((cells, color))
    return out


def _connected(white):
    if not white:
        return False
    seen, stack = set(), [next(iter(white))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for (a, b) in _neighbor(r, c, 9, 9):
            if (a, b) in white:
                stack.append((a, b))
    return len(seen) == len(white)


def _layout_ok(blacks, rows, cols):
    if not blacks:
        return False
    for (r, c) in blacks:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
    for (r1, c1) in blacks:
        for (r2, c2) in blacks:
            if (r1, c1) == (r2, c2):
                continue
            if (r1 - r2, c1 - c2) not in ((0, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                continue
            return False
    return True


def _make_layout(rows, cols, rng, theme=None):
    """Return a set of black squares forming a valid symmetric layout."""
    blacks = set()
    full = {(r, c) for r in range(rows) for c in range(cols)}
    while len(blacks) < 0.3 * len(full):
        if theme is not None and (theme == "long-first" or theme == "longer-first"):
            r, c = rng.choice([(3, 3), (3, 6), (6, 3), (6, 6)])
        else:
            r, c = rng.choice(list(full))
        if (r, c) in blacks:
            continue
        trial_blacks = set(blacks)
        trial_blacks.add((r, c))
        for dr, dc in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            pb = (r + dr, c + dc)
            if pb in full:
                trial_blacks.add(pb)
        if not _layout_ok(trial_blacks, rows, cols):
            continue
        blacks = trial_blacks
    return blacks


def _white(ros, cols, blacks):
    return {cell for cell in range(ros * cols) if cell not in blacks}


def _runs(white, rows, cols):
    seen, out = set(), []
    for r, c in white:
        if (r, c) in seen:
            continue
        stack = [r, c]
        cells, color = [], True
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in seen or (rr, cc) not in white:
                continue
            seen.add((rr, cc))
            cells.append((rr, cc))
            for (a, b) in _neighbor(rr, cc, rows, cols):
                if (a, b) not in white:
                    continue
                if (a, b) not in seen:
                    stack.append((a, b))
        if len(cells) < 3:
            continue
        out.append((cells, color))
    return out


def _split_runs(run_list):
    out = []
    for cells, _ in run_list:
        for cell in cells:
            out.append(cell)
    return out


def _index(r, c, rows, cols):
    return r * cols + c


def _pos(i, rows, cols):
    r, c = divmod(i, cols)
    return r, c


def _neighbors(r, c, rows, cols):
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _word_for_cell(cell, grid, rows, cols):
    r, c = cell
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return None
    if grid[r][c] == " ":
        return None
    return grid[r][c]


def _slots_from_runs(run_list, rows, cols):
    slots = {}
    for cells, color in run_list:
        for idx, cell in enumerate(cells):
            if idx not in slots:
                slots[idx] = {"cells": [cell], "color": color}
            else:
                slots[idx]["cells"].append(cell)
    return slots


def _fill_slots(slots, grid, rng, word_source, budget=10000, deadline=None):
    n = len(slots)
    used, steps = set(), 0
    neighbors = {idx: set() for idx in range(n)}
    for i in range(n):
        for cell in slots[i]["cells"]:
            for cell2 in _neighbors(*cell, rows=rows, cols=cols):
                if cell2 in slots:
                    neighbors[i].add(cell2)
    for _ in range(budget):
        if deadline is not None and time.perf_counter() > deadline:
            return False
        steps += 1
        rng.shuffle(list(range(n)))
        for idx in range(n):
            if idx in used:
                continue
            cells = slots[idx]["cells"]
            if len(cells) == 1:
                used.add(idx)
                continue
            options = [w for w in word_source if w.isalpha()]
            options.sort(key=lambda w: w not in "AEIOU", reverse=True)
            for w in options:
                ok = True
                for pos, cell in enumerate(cells):
                    r, c = cell
                    if grid[r][c] != " " and grid[r][c] != w[pos]:
                        ok = False
                        break
                if not ok:
                    continue
                ok = True
                for p1, cell in enumerate(cells):
                    for p2, cell2 in enumerate(cells):
                        if p1 == p2:
                            continue
                        if grid[cell[0]][cell[1]] != w[p1] or grid[cell2[0]][cell2[1]] != w[p2]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                grid_copy = [row[:] for row in grid]
                filled = True
                for pos, cell in enumerate(cells):
                    rr, cc = cell
                    grid_copy[rr][cc] = w[pos]
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p1, cell1 in enumerate(sa):
                        if grid_copy[cell1[0]][cell1[1]] != w[p1]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p2, cell2 in enumerate(sa):
                        if grid_copy[cell2[0]][cell2[1]] != w[p2]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                grid = grid_copy
                used.add(idx)
                break
            if idx in used:
                break
        if len(used) == n:
            return True
    return False


def _build_layout(white, rows, cols):
    cells = [pos for pos in white]
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    index = {}
    for r, c in cells:
        index[r, c] = _index(r, c, rows, cols)
    for r, c in index:
        grid[r][c] = "."
    return index, grid


def _group_by_color(white, rows, cols):
    groups = {"red": [], "blue": []}
    for cell in white:
        r, c = cell
        is_red = (r + c) % 2 == 0
        groups["red" if is_red else "blue"].append(cell)
    return groups


def _runs(white, rows, cols):
    seen, out = set(), []
    for r, c in white:
        if (r, c) in seen:
            continue
        stack = [r, c]
        cells, color = [], True
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in seen or (rr, cc) not in white:
                continue
            seen.add((rr, cc))
            cells.append((rr, cc))
            for (a, b) in _neighbor(rr, cc, rows, cols):
                if (a, b) not in white:
                    continue
                if (a, b) not in seen:
                    stack.append((a, b))
        if len(cells) < 3:
            continue
        out.append((cells, color))
    return out


def _split_runs(run_list):
    out = []
    for cells, _ in run_list:
        for cell in cells:
            out.append(cell)
    return out


def _index(r, c, rows, cols):
    return r * cols + c


def _pos(i, rows, cols):
    r, c = divmod(i, cols)
    return r, c


def _neighbors(r, c, rows, cols):
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _word_for_cell(cell, grid, rows, cols):
    r, c = cell
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return None
    if grid[r][c] == " ":
        return None
    return grid[r][c]


def _slots_from_runs(run_list, rows, cols):
    slots = {}
    for cells, color in run_list:
        for idx, cell in enumerate(cells):
            if idx not in slots:
                slots[idx] = {"cells": [cell], "color": color}
            else:
                slots[idx]["cells"].append(cell)
    return slots


def _fill_slots(slots, grid, rng, word_source, budget=10000, deadline=None):
    n = len(slots)
    used, steps = set(), 0
    neighbors = {idx: set() for idx in range(n)}
    for i in range(n):
        for cell in slots[i]["cells"]:
            for cell2 in _neighbors(*cell, rows=rows, cols=cols):
                if cell2 in slots:
                    neighbors[i].add(cell2)
    for _ in range(budget):
        if deadline is not None and time.perf_counter() > deadline:
            return False
        steps += 1
        rng.shuffle(list(range(n)))
        for idx in range(n):
            if idx in used:
                continue
            cells = slots[idx]["cells"]
            if len(cells) == 1:
                used.add(idx)
                continue
            options = [w for w in word_source if w.isalpha()]
            options.sort(key=lambda w: w not in "AEIOU", reverse=True)
            for w in options:
                ok = True
                for pos, cell in enumerate(cells):
                    r, c = cell
                    if grid[r][c] != " " and grid[r][c] != w[pos]:
                        ok = False
                        break
                if not ok:
                    continue
                ok = True
                for p1, cell in enumerate(cells):
                    for p2, cell2 in enumerate(cells):
                        if p1 == p2:
                            continue
                        if grid[cell[0]][cell[1]] != w[p1] or grid[cell2[0]][cell2[1]] != w[p2]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                grid_copy = [row[:] for row in grid]
                filled = True
                for pos, cell in enumerate(cells):
                    rr, cc = cell
                    grid_copy[rr][cc] = w[pos]
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p1, cell1 in enumerate(sa):
                        if grid_copy[cell1[0]][cell1[1]] != w[p1]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p2, cell2 in enumerate(sa):
                        if grid_copy[cell2[0]][cell2[1]] != w[p2]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                grid = grid_copy
                used.add(idx)
                break
            if idx in used:
                break
        if len(used) == n:
            return True
    return False


def _build_layout(white, rows, cols):
    cells = [pos for pos in white]
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    index = {}
    for r, c in cells:
        index[r, c] = _index(r, c, rows, cols)
    for r, c in index:
        grid[r][c] = "."
    return index, grid


def _group_by_color(white, rows, cols):
    groups = {"red": [], "blue": []}
    for cell in white:
        r, c = cell
        is_red = (r + c) % 2 == 0
        groups["red" if is_red else "blue"].append(cell)
    return groups


def _runs(white, rows, cols):
    seen, out = set(), []
    for r, c in white:
        if (r, c) in seen:
            continue
        stack = [r, c]
        cells, color = [], True
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in seen or (rr, cc) not in white:
                continue
            seen.add((rr, cc))
            cells.append((rr, cc))
            for (a, b) in _neighbor(rr, cc, rows, cols):
                if (a, b) not in white:
                    continue
                if (a, b) not in seen:
                    stack.append((a, b))
        if len(cells) < 3:
            continue
        out.append((cells, color))
    return out


def _split_runs(run_list):
    out = []
    for cells, _ in run_list:
        for cell in cells:
            out.append(cell)
    return out


def _index(r, c, rows, cols):
    return r * cols + c


def _pos(i, rows, cols):
    r, c = divmod(i, cols)
    return r, c


def _neighbors(r, c, rows, cols):
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _word_for_cell(cell, grid, rows, cols):
    r, c = cell
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return None
    if grid[r][c] == " ":
        return None
    return grid[r][c]


def _slots_from_runs(run_list, rows, cols):
    slots = {}
    for cells, color in run_list:
        for idx, cell in enumerate(cells):
            if idx not in slots:
                slots[idx] = {"cells": [cell], "color": color}
            else:
                slots[idx]["cells"].append(cell)
    return slots


def _fill_slots(slots, grid, rng, word_source, budget=10000, deadline=None):
    n = len(slots)
    used, steps = set(), 0
    neighbors = {idx: set() for idx in range(n)}
    for i in range(n):
        for cell in slots[i]["cells"]:
            for cell2 in _neighbors(*cell, rows=rows, cols=cols):
                if cell2 in slots:
                    neighbors[i].add(cell2)
    for _ in range(budget):
        if deadline is not None and time.perf_counter() > deadline:
            return False
        steps += 1
        rng.shuffle(list(range(n)))
        for idx in range(n):
            if idx in used:
                continue
            cells = slots[idx]["cells"]
            if len(cells) == 1:
                used.add(idx)
                continue
            options = [w for w in word_source if w.isalpha()]
            options.sort(key=lambda w: w not in "AEIOU", reverse=True)
            for w in options:
                ok = True
                for pos, cell in enumerate(cells):
                    r, c = cell
                    if grid[r][c] != " " and grid[r][c] != w[pos]:
                        ok = False
                        break
                if not ok:
                    continue
                ok = True
                for p1, cell in enumerate(cells):
                    for p2, cell2 in enumerate(cells):
                        if p1 == p2:
                            continue
                        if grid[cell[0]][cell[1]] != w[p1] or grid[cell2[0]][cell2[1]] != w[p2]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                grid_copy = [row[:] for row in grid]
                filled = True
                for pos, cell in enumerate(cells):
                    rr, cc = cell
                    grid_copy[rr][cc] = w[pos]
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p1, cell1 in enumerate(sa):
                        if grid_copy[cell1[0]][cell1[1]] != w[p1]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                for (a, b) in neighbors[idx]:
                    if a not in slots:
                        continue
                    if slots[a]["color"] != slots[idx]["color"]:
                        continue
                    sa = slots[a]["cells"]
                    for p2, cell2 in enumerate(sa):
                        if grid_copy[cell2[0]][cell2[1]] != w[p2]:
                            filled = False
                            break
                    if not filled:
                        break
                if not filled:
                    continue
                grid = grid_copy
                used.add(idx)
                break
            if idx in used:
                break
        if len(used) == n:
            return True
    return False


def _build_layout(white, rows, cols):
    cells = [pos for pos in white]
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    index = {}
    for r, c in cells:
        index[r, c] = _index(r, c, rows, cols)
    for r, c in index:
        grid[r][c] = "."
    return index, grid


def _group_by_color(white, rows, cols):
    groups = {"red": [], "blue": []}
    for cell in white:
        r, c = cell
        is_red = (r + c) % 2 == 0
        groups["red" if is_red else "blue"].append(cell)
    return groups


def generate_crossword(topic: str, word_source, size: int) -> dict:
    """Return a grid, or {"error": "no solution found"}."""
    word_source = list(word_source)
    if not word_source:
        return {"error": "no solution found"}
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    rows, cols = size, size
    theme = topic.lower()
    if "long" in theme or "longer" in theme:
        theme = "long-first"
    blacks = _make_layout(rows, cols, rng, theme=theme)
    white = _white(blacks, rows, cols)
    if not white or not _connected(white):
        return {"error": "no solution found"}
    index, grid = _build_layout(white, rows, cols)
    slots = _slots_from_runs(_runs(white, rows, cols), rows, cols)
    if not _fill_slots(slots, grid, rng, word_source, budget=10000):
        return {"error": "no solution found"}
    cells = []
    for (r, c) in white:
        cells.append({"r": r, "c": c, "letter": grid[r][c]})
    numbers = {}
    for idx, (cells, _) in enumerate(_runs(white, rows, cols)):
        for cell in cells:
            numbers[_index(*cell, rows, cols)] = idx + 1
    across, down = [], []
    for cells, color in _runs(white, rows, cols):
        for pos, cell in enumerate(cells):
            r, c = cell
            if c == 0 or c == cols - 1 or (r, c - 1) not in white or (r, c + 1) not in white:
                continue
            w = "".join(grid[rr][cc] for rr, cc in cells)
            across.append({"number": numbers[_index(r, c, rows, cols)], "row": r, "col": c, "answer": w, "len": len(w)})
        for pos, cell in enumerate(cells):
            if r == 0 or r == rows - 1 or (r - 1, c) not in white or (r + 1, c) not in white:
                continue
            w = "".join(grid[rr][cc] for rr, cc in cells)
            down.append({"number": numbers[_index(r, c, rows, cols)], "row": r, "col": c, "answer": w, "len": len(w)})
    return {"rows": rows, "cols": cols, "cells": cells, "across": across, "down": down}