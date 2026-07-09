def generate_crossword(topic: str, word_source, size: int) -> dict:
    import random
    from collections import defaultdict

    rng = random.Random(1234567 + size)
    N = size

    def clean(ws):
        out = []
        seen = set()
        for w in ws:
            if not isinstance(w, str):
                continue
            u = w.upper()
            if u.isalpha() and u not in seen:
                seen.add(u)
                out.append(u)
        return out

    theme_words = clean(word_source.get("theme", []))
    fill_words = clean(word_source.get("fill", []))
    all_words = []
    seen = set()
    for w in theme_words + fill_words:
        if w not in seen:
            seen.add(w)
            all_words.append(w)

    words_by_len = defaultdict(list)
    for w in all_words:
        words_by_len[len(w)].append(w)

    def compute_slots(black):
        across, down = [], []
        for r in range(N):
            c = 0
            while c < N:
                if (r, c) in black:
                    c += 1
                    continue
                start = c
                while c < N and (r, c) not in black:
                    c += 1
                across.append((r, start, c - start))
        for c in range(N):
            r = 0
            while r < N:
                if (r, c) in black:
                    r += 1
                    continue
                start = r
                while r < N and (r, c) not in black:
                    r += 1
                down.append((start, c, r - start))
        return across, down

    def validate_pattern(black):
        across, down = compute_slots(black)
        cover_a, cover_d = {}, {}
        for (r, c, l) in across:
            if l < 3:
                return False, None, None
            for i in range(l):
                cover_a[(r, c + i)] = True
        for (r, c, l) in down:
            if l < 3:
                return False, None, None
            for i in range(l):
                cover_d[(r + i, c)] = True
        white = [(r, c) for r in range(N) for c in range(N) if (r, c) not in black]
        if not white:
            return False, None, None
        for cell in white:
            if cell not in cover_a or cell not in cover_d:
                return False, None, None
        stack = [white[0]]
        seen_c = {white[0]}
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc = (r + dr, c + dc)
                if 0 <= nc[0] < N and 0 <= nc[1] < N and nc not in black and nc not in seen_c:
                    seen_c.add(nc)
                    stack.append(nc)
        if len(seen_c) != len(white):
            return False, None, None
        return True, across, down

    def gen_pattern(ratio):
        cells = [(r, c) for r in range(N) for c in range(N)]
        pairs, done = [], set()
        for (r, c) in cells:
            if (r, c) in done:
                continue
            r2, c2 = N - 1 - r, N - 1 - c
            done.add((r, c))
            done.add((r2, c2))
            pairs.append(((r, c), (r2, c2)))
        rng.shuffle(pairs)
        target = int(N * N * ratio)
        black = set()
        for p1, p2 in pairs:
            if len(black) >= target:
                break
            black.add(p1)
            black.add(p2)
        return black

    def attempt_build():
        ratios = [0.0, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3] if N >= 9 else [0.0, 0.05, 0.1, 0.15, 0.2]
        for ratio in ratios:
            for _ in range(80):
                black = gen_pattern(ratio)
                ok, across, down = validate_pattern(black)
                if ok:
                    return black, across, down
        black = set()
        ok, across, down = validate_pattern(black)
        if ok:
            return black, across, down
        return set(), [], []

    black, across_slots, down_slots = attempt_build()

    slots = []
    for (r, c, l) in across_slots:
        slots.append({"dir": "A", "r": r, "c": c, "len": l, "cells": [(r, c + i) for i in range(l)]})
    for (r, c, l) in down_slots:
        slots.append({"dir": "D", "r": r, "c": c, "len": l, "cells": [(r + i, c) for i in range(l)]})

    grid_letters = {}
    assignment = {}
    used_words = set()

    def candidates_for(idx):
        s = slots[idx]
        pattern = [grid_letters.get(cell) for cell in s["cells"]]
        cands = []
        for w in words_by_len.get(s["len"], []):
            if w in used_words:
                continue
            ok = True
            for i, ch in enumerate(pattern):
                if ch is not None and w[i] != ch:
                    ok = False
                    break
            if ok:
                cands.append(w)
        return cands

    def pick_next(unassigned):
        best_idx, best_cands, best_n = None, None, None
        for idx in unassigned:
            cands = candidates_for(idx)
            if len(cands) == 0:
                return idx, []
            if best_n is None or len(cands) < best_n:
                best_n = len(cands)
                best_idx = idx
                best_cands = cands
        return best_idx, best_cands if best_cands is not None else []

    unassigned = set(range(len(slots)))
    steps = [400000]

    def backtrack():
        if not unassigned:
            return True
        steps[0] -= 1
        if steps[0] <= 0:
            return False
        idx, cands = pick_next(unassigned)
        if idx is None or not cands:
            return False
        cands = list(cands)
        rng.shuffle(cands)
        cands.sort(key=lambda w: 0 if w in theme_words else 1)
        s = slots[idx]
        for w in cands:
            conflict = False
            for i, cell in enumerate(s["cells"]):
                if cell in grid_letters and grid_letters[cell] != w[i]:
                    conflict = True
                    break
            if conflict:
                continue
            added = []
            for i, cell in enumerate(s["cells"]):
                if cell not in grid_letters:
                    grid_letters[cell] = w[i]
                    added.append(cell)
            assignment[idx] = w
            used_words.add(w)
            unassigned.discard(idx)

            if backtrack():
                return True

            unassigned.add(idx)
            used_words.discard(w)
            del assignment[idx]
            for cell in added:
                del grid_letters[cell]
        return False

    success = backtrack() if slots else True

    if not success:
        grid_letters.clear()
        assignment.clear()
        used_words.clear()
        order = sorted(range(len(slots)), key=lambda i: -slots[i]["len"])
        for idx in order:
            cands = candidates_for(idx)
            if not cands:
                cands = list(words_by_len.get(slots[idx]["len"], []))
            if cands:
                w = cands[0]
                s = slots[idx]
                ok = True
                for i, cell in enumerate(s["cells"]):
                    if cell in grid_letters and grid_letters[cell] != w[i]:
                        ok = False
                        break
                if ok:
                    for i, cell in enumerate(s["cells"]):
                        grid_letters[cell] = w[i]
                    assignment[idx] = w
                    used_words.add(w)

    starts = {}
    for idx, s in enumerate(slots):
        starts[s["cells"][0]] = True

    numbers = {}
    counter = 1
    for r in range(N):
        for c in range(N):
            if (r, c) in starts:
                numbers[(r, c)] = counter
                counter += 1

    cells_out = []
    for r in range(N):
        for c in range(N):
            cell = (r, c)
            if cell in grid_letters:
                entry = {"r": r, "c": c, "letter": grid_letters[cell]}
                if cell in numbers:
                    entry["number"] = numbers[cell]
                cells_out.append(entry)

    across_out, down_out = [], []
    for idx, s in enumerate(slots):
        cell0 = s["cells"][0]
        num = numbers.get(cell0)
        answer = assignment.get(idx)
        if answer is None:
            answer = "".join(grid_letters.get(cell, "?") for cell in s["cells"])
        entry = {"number": num, "row": s["r"], "col": s["c"], "answer": answer, "len": s["len"]}
        if s["dir"] == "A":
            across_out.append(entry)
        else:
            down_out.append(entry)

    across_out.sort(key=lambda e: e["number"] if e["number"] is not None else 999999)
    down_out.sort(key=lambda e: e["number"] if e["number"] is not None else 999999)

    return {
        "rows": N,
        "cols": N,
        "cells": cells_out,
        "across": across_out,
        "down": down_out
    }