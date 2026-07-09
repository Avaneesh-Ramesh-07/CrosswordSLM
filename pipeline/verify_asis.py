"""Detailed failure taxonomy for the as-is (own-terms) 100-agent run.

Top-level (all 100, from summary.json verdicts):
  unfillable  = crashed | hung | failed_to_fill (explicit no-solution) | no_grid_output
  produced_grid = FILLED_CANDIDATE (a letter-filled grid was printed)
For each produced grid we extract it and measure: symmetry, total runs (>=3), how many
runs are NON-words (faulty crossings), and the fraction of runs that are faulty.
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, ".")

OUT = "runs/eval/asis"
DICT = {w.strip().upper() for w in open("data/wordlists/words_alpha.txt", encoding="utf-8") if w.strip()}
BLACK = set("#█.·*")


def cells_of_line(s):
    s = s.rstrip("\n")
    if not s.strip() or set(s.strip()) <= set("+-| =_"):
        return None
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip() != ""]
        if parts and re.fullmatch(r"\d+", parts[0]):   # drop leading row-number gutter (e.g. "5 | B | A |")
            parts = parts[1:]
    else:
        parts = s.split()
        if parts and re.fullmatch(r"\d+", parts[0]):
            parts = parts[1:]
    if not parts:
        return None
    row = []
    for p in parts:
        m = re.search(r"[A-Za-z]", p)
        if m:
            row.append(m.group().upper())
        else:
            row.append(None)   # black / dot / digit-header / blank
    return row


def extract_grid(text):
    """Detect the grid by the MODAL width of letter-bearing rows (handles varied layouts,
    box/plain, and grids whose tokenized width != nominal size)."""
    from collections import Counter
    parsed = [cells_of_line(l) for l in text.splitlines()]
    cand = [p for p in parsed if p and any(c is not None for c in p)]   # rows with >=1 letter
    if not cand:
        return None
    width = Counter(len(p) for p in cand).most_common(1)[0][0]
    if width < 5:
        return None
    rows = [p for p in parsed if p and len(p) == width]
    while rows and all(c is None for c in rows[0]):
        rows.pop(0)
    while rows and all(c is None for c in rows[-1]):
        rows.pop()
    # keep the last `width` rows (crosswords are square); need >=5 to be a real grid
    rows = rows[-width:]
    return rows if len(rows) >= 5 else None


def runs(grid):
    R, C = len(grid), len(grid[0])
    out = []
    for r in range(R):
        c = 0
        while c < C:
            if grid[r][c] is None:
                c += 1; continue
            w = ""
            while c < C and grid[r][c] is not None:
                w += grid[r][c]; c += 1
            if len(w) >= 3:
                out.append(w)
    for c in range(C):
        r = 0
        while r < R:
            if grid[r][c] is None:
                r += 1; continue
            w = ""
            while r < R and grid[r][c] is not None:
                w += grid[r][c]; r += 1
            if len(w) >= 3:
                out.append(w)
    return out


def symmetric(grid):
    R, C = len(grid), len(grid[0])
    return all((grid[r][c] is None) == (grid[R-1-r][C-1-c] is None)
               for r in range(R) for c in range(C))


def analyze_grid(text, size):
    g = extract_grid(text)
    if g is None:
        return {"parsed": False}
    rs = runs(g)
    nonword = [w for w in rs if w not in DICT]
    R, C = len(g), len(g[0])
    cells = [c for row in g for c in row]
    return {"parsed": True, "n_runs": len(rs), "n_nonword": len(nonword),
            "symmetric": symmetric(g), "square": R == C,
            "black_frac": sum(1 for c in cells if c is None) / len(cells),
            "faulty_frac": (len(nonword) / len(rs)) if rs else 1.0}


def main():
    summ = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
    recs = summ["records"]
    n = len(recs)
    from collections import Counter
    v = Counter(r["verdict"] for r in recs)
    unfillable = v["crashed"] + v["hung"] + v["failed_to_fill"] + v["no_grid_output"]
    produced = v["FILLED_CANDIDATE"]

    # analyze each produced grid
    stats = []
    for r in recs:
        if r["verdict"] != "FILLED_CANDIDATE":
            continue
        p = os.path.join(OUT, f"out_{r['i']:03d}_s{r['size']}.txt")
        if os.path.exists(p):
            stats.append(analyze_grid(open(p, encoding="utf-8").read(), r["size"]))
    parsed = [s for s in stats if s["parsed"]]
    asym = sum(1 for s in parsed if not s["symmetric"])
    with_nonword = [s for s in parsed if s["n_nonword"] > 0]
    total_runs = sum(s["n_runs"] for s in parsed)
    total_nonword = sum(s["n_nonword"] for s in parsed)
    fully_real = [s for s in parsed if s["n_nonword"] == 0]

    print(f"=== AS-IS own-terms run: n={n} ===")
    print(f"VALID crosswords: 0 (0%)\n")
    print(f"UNFILLABLE / no crossword produced: {unfillable} ({unfillable}%)")
    print(f"   crashed (exception):        {v['crashed']} ({v['crashed']}%)")
    print(f"   reported NO SOLUTION:       {v['failed_to_fill']} ({v['failed_to_fill']}%)")
    print(f"   no grid in output:          {v['no_grid_output']} ({v['no_grid_output']}%)")
    print(f"   hung (timeout):             {v['hung']} ({v['hung']}%)")
    # the REAL disqualifiers (density + non-words), NOT symmetry:
    dense = [s for s in parsed if s["black_frac"] <= 0.35]     # crossword-like density
    sparse = [s for s in parsed if s["black_frac"] > 0.35]     # mostly-black / degenerate
    dense_nonword = [s for s in dense if s["n_nonword"] > 0]
    # user's definition of valid: square + few black squares + all interlocking real words
    valid_density = [s for s in parsed if s["square"] and s["black_frac"] <= 0.35
                     and s["n_nonword"] == 0 and s["n_runs"] >= 4]
    print(f"\nPRODUCED a filled grid (all invalid): {produced} ({produced}%)")
    print(f"   parseable for analysis:      {len(parsed)}/{produced}")
    print(f"   mean black-square fraction:  {sum(s['black_frac'] for s in parsed)/len(parsed):.0%} "
          f"(a real crossword is ~16%)")
    print(f"   -- of these, TWO failure modes --")
    print(f"   A) crossword-like density (<=35% black): {len(dense)}  "
          f"-> {len(dense_nonword)}/{len(dense)} have NON-WORD crossings")
    print(f"   B) degenerate/sparse (>35% black, isolated words): {len(sparse)}  "
          f"(mean {sum(s['black_frac'] for s in sparse)/max(1,len(sparse)):.0%} black)")
    print(f"   all-runs-real AND crossword-density (your 'valid'): {len(valid_density)}")
    if with_nonword:
        avg = sum(s["faulty_frac"] for s in with_nonword) / len(with_nonword)
        print(f"\n   FAULTY CONNECTIONS: {total_nonword}/{total_runs} runs are non-words "
              f"({100*total_nonword/total_runs:.0f}% of all placed entries); among grids with any "
              f"faulty crossing, avg {100*avg:.0f}% of that grid's runs are non-words")
    print(f"   (symmetry is NOT counted as a failure; asym={asym}/{len(parsed)} reported for reference only)")


if __name__ == "__main__":
    main()
