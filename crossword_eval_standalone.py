#!/usr/bin/env python3
"""Self-contained crossword eval (NO repo needed).

    pip install wordfreq pyspellchecker
    python crossword_eval_standalone.py --lang en --submission submission_en.py
    python crossword_eval_standalone.py --lang es --submission submission_es.py

Builds a clean palette (wordfreq INTERSECT a real dictionary), runs the submitted
generate_crossword(topic, word_source, size) across sizes/seeds, and validates
EVERY entry against the real dictionary -- palette membership alone is not trusted,
since raw frequency lists contain acronyms/proper nouns. Each run is hard-capped by
a per-size time budget.
"""
import argparse
import importlib.util
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

BUDGET = {7: 3.0, 9: 5.0, 11: 12.0, 13: 20.0, 15: 30.0}
TOPICS = {"en": ["vocabulary", "words", "study", "learn", "review", "practice", "school", "language"],
          "es": ["vocabulario", "palabras", "estudio", "aprender", "repaso", "escuela", "idioma", "examen"]}
STOP = {"MIERDA", "PUTA", "PUTO", "JODER", "CONO", "POLLA", "CABRON", "MARICON", "EYACULACION",
        "PENE", "CULO", "TETAS", "COJONES", "FUCK", "SHIT", "CUNT", "COCK"}


def norm(w):
    w = unicodedata.normalize("NFKD", str(w))
    return "".join(c for c in w if not unicodedata.combining(c)).upper()


def build_palette(lang, max_len, freq_n=60000, min_len=3):
    import wordfreq
    from spellchecker import SpellChecker
    DICT = {norm(w) for w in SpellChecker(language=lang).word_frequency.dictionary}
    seen, ordered = set(), []
    for w in wordfreq.top_n_list(lang, freq_n):
        u = norm(w)
        if u.isalpha() and min_len <= len(u) <= max_len and u not in seen and u in DICT and u not in STOP:
            seen.add(u); ordered.append(u)
    theme = [w for w in ordered if len(w) >= 4]
    tset = set(theme)
    return {"theme": theme, "fill": [w for w in ordered if w not in tset],
            "allowed": set(ordered), "DICT": DICT}


def _conn(white):
    if not white:
        return False
    s = next(iter(white)); seen = {s}; st = [s]
    while st:
        r, c = st.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in white and nb not in seen:
                seen.add(nb); st.append(nb)
    return len(seen) == len(white)


def _sym(white, size):
    return all(((r, c) in white) == ((size - 1 - r, size - 1 - c) in white)
               for r in range(size) for c in range(size))


def score(layout, size, allowed, theme, DICT, min_len=3):
    R = {"valid": 0, "coverage": 0.0, "filler": 0.0, "dict_frac": 0.0, "crossings": 0, "n_entries": 0}
    if not isinstance(layout, dict) or "across" not in layout or "down" not in layout:
        return R
    across, down = layout.get("across") or [], layout.get("down") or []
    grid = {}; conflict = oob = False

    def place(word, r, c, dr, dc):
        nonlocal conflict, oob
        for i, ch in enumerate(word):
            rr, cc = r + dr * i, c + dc * i
            if not (0 <= rr < size and 0 <= cc < size):
                oob = True; return
            if (rr, cc) in grid and grid[(rr, cc)] != ch:
                conflict = True
            grid[(rr, cc)] = ch
    try:
        for e in across:
            place(norm(e["answer"]), int(e["row"]), int(e["col"]), 0, 1)
        for e in down:
            place(norm(e["answer"]), int(e["row"]), int(e["col"]), 1, 0)
    except Exception:
        return R
    white = set(grid)
    if not white:
        return R

    def runs(dr, dc):
        out = []
        for (r, c) in white:
            if (r - dr, c - dc) in white:
                continue
            w = ""; rr, cc = r, c; L = 0
            while (rr, cc) in white:
                w += grid[(rr, cc)]; L += 1; rr += dr; cc += dc
            out.append((r, c, w, L))
        return out
    hr, vr = runs(0, 1), runs(1, 0)
    bad_short = [x for x in hr + vr if x[3] < min_len]
    actual_a = {(r, c, w) for (r, c, w, l) in hr}
    actual_d = {(r, c, w) for (r, c, w, l) in vr}
    claimed_a = {(int(e["row"]), int(e["col"]), norm(e["answer"])) for e in across}
    claimed_d = {(int(e["row"]), int(e["col"]), norm(e["answer"])) for e in down}
    answers = [w for (r, c, w, l) in hr + vr if l >= min_len]
    nonword = [w for w in answers if w not in allowed]
    valid = (not conflict and not oob and not bad_short and actual_a == claimed_a
             and actual_d == claimed_d and not nonword and _conn(white) and _sym(white, size)
             and layout.get("rows") == size and layout.get("cols") == size)
    ac = {(r, c + i) for (r, c, w, l) in hr for i in range(l)}
    dn = {(r + i, c) for (r, c, w, l) in vr for i in range(l)}
    R["valid"] = 1 if valid else 0
    R["n_entries"] = len(across) + len(down)
    R["crossings"] = len(ac & dn)
    if answers:
        R["coverage"] = sum(1 for w in answers if w in theme) / len(answers)
        R["filler"] = sum(1 for w in answers if w not in allowed) / len(answers)
        R["dict_frac"] = sum(1 for w in answers if w in DICT) / len(answers)
    return R


def run_one(fn, topic, ws, size, timeout):
    """Run generate_crossword with a hard time cap; None if it exceeds it or errors."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, topic, ws, size)
        try:
            return fut.result(timeout=timeout)
        except FTimeout:
            return None
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "es"], required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--per-size", type=int, default=8)
    a = ap.parse_args()
    sizes = ([int(s) for s in a.sizes.split(",")] if a.sizes
             else ([7, 9, 11, 15] if a.lang == "en" else [7, 9, 11]))
    print("building palette + dictionary (wordfreq + pyspellchecker)...")
    pal = build_palette(a.lang, max(sizes))
    print(f"palette {len(pal['allowed'])} words | dictionary {len(pal['DICT'])} words")
    spec = importlib.util.spec_from_file_location("sub", a.submission)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not hasattr(mod, "generate_crossword"):
        sys.exit("submission has no generate_crossword(topic, word_source, size)")
    ws = {"theme": pal["theme"], "fill": pal["fill"]}
    tset = set(pal["theme"])
    rows = []
    for size in sizes:
        budget = BUDGET.get(size, size * 2)
        for t in TOPICS[a.lang][:a.per_size]:
            t0 = time.perf_counter()
            lay = run_one(mod.generate_crossword, t, ws, size, budget * 2 + 3)
            dt = time.perf_counter() - t0
            if lay is None:
                rows.append({"size": size, "valid": 0, "within": 0, "fully": 0, "dict_frac": 0,
                             "coverage": 0, "crossings": 0, "entries": 0, "filler": 0, "rt": round(dt, 2)})
                continue
            m = score(lay, size, pal["allowed"], tset, pal["DICT"])
            within = int(m["valid"] and m["filler"] <= 0.30 and dt <= budget)
            rows.append({"size": size, "valid": m["valid"], "within": within,
                         "fully": int(m["valid"] and m["dict_frac"] >= 0.999), "dict_frac": m["dict_frac"],
                         "coverage": m["coverage"], "crossings": m["crossings"], "entries": m["n_entries"],
                         "filler": m["filler"], "rt": round(dt, 2)})

    def agg(rs):
        n = len(rs) or 1
        v = [r for r in rs if r["valid"]]; vn = len(v) or 1
        return (len(rs), sum(r["valid"] for r in rs) / n, sum(r["fully"] for r in rs) / n,
                sum(r["within"] for r in rs) / n, sum(r["dict_frac"] for r in rs) / n,
                sum(r["coverage"] for r in v) / vn, sum(r["crossings"] for r in v) / vn,
                sum(r["entries"] for r in v) / vn, sum(r["filler"] for r in v) / vn,
                sum(r["rt"] for r in rs) / n)

    hdr = (f"{'size':>5}{'n':>4}{'valid%':>8}{'fullyOK%':>10}{'within%':>9}"
           f"{'dictOK':>8}{'cov':>6}{'cross':>7}{'entries':>8}{'filler%':>9}{'rt':>7}")
    print("\n" + hdr); print("-" * len(hdr))
    for size in sizes:
        n, val, ful, wit, dic, cov, cr, en, fil, rt = agg([r for r in rows if r["size"] == size])
        print(f"{size:>5}{n:>4}{val*100:>7.0f}{ful*100:>9.0f}{wit*100:>8.0f}{dic*100:>7.0f}"
              f"{cov:>6.2f}{cr:>7.0f}{en:>8.0f}{fil*100:>8.0f}{rt:>7.2f}")
    n, val, ful, wit, dic, cov, cr, en, fil, rt = agg(rows); print("-" * len(hdr))
    print(f"{'ALL':>5}{n:>4}{val*100:>7.0f}{ful*100:>9.0f}{wit*100:>8.0f}{dic*100:>7.0f}"
          f"{cov:>6.2f}{cr:>7.0f}{en:>8.0f}{fil*100:>8.0f}{rt:>7.2f}")
    print("\nfullyOK% = structurally valid AND every entry a real dictionary word")


if __name__ == "__main__":
    main()
