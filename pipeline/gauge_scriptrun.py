"""Run-as-script gauge (Option B): evaluate a crossword program by actually running it.

For each program it does `python <prog.py>` (a fresh process each time, so the per-process
`hash((topic,size))` seed differs -> a different random structure per run), captures stdout,
parses the printed grid + Across/Down clues, reconstructs the crossword, and scores it with the
established criteria (judged against a real dictionary; symmetry NOT required):
  exactly size x size; every white run (across & down) >= 3 letters; declared clues == the
  grid's actual maximal runs; all white cells connected; every entry a real dictionary word.
This is NOT a harness that calls generate_crossword -- it judges what the standalone program
actually prints.

    python pipeline/gauge_scriptrun.py <dir> --repeats 3 --workers 4 [--timeout 40] [--out f.json]

Size per program comes from the `_sNN` filename suffix.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.gauge_selfcontained import load_dict


def run_script(path, timeout):
    """Run `python path` in a fresh process; return (stdout, err_or_None)."""
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, f"launch-error: {type(e).__name__}: {e}"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return None, "crash: " + (tail[-1][:80] if tail else f"exit {p.returncode}")
    return p.stdout, None


def validate_printed(out, size, DICT):
    """Parse the printed grid + clues and score the crossword. Returns {valid, reason}."""
    res = {"valid": False, "reason": ""}
    lines = out.splitlines()

    # grid = leading rows of single-char space-separated tokens, until blank / "Across"/"Down"
    grid = []
    for ln in lines:
        s = ln.strip()
        if s == "" or s.startswith(("Across", "Down")):
            break
        toks = ln.split()
        if toks and all(len(t) == 1 for t in toks):
            grid.append([t.upper() for t in toks])
        else:
            break
    if not grid:
        res["reason"] = "no grid printed"; return res
    n = len(grid)
    if any(len(r) != n for r in grid):
        res["reason"] = "ragged/non-square grid"; return res
    if size and n != size:
        res["reason"] = f"printed {n}x{n}, expected {size}x{size}"; return res

    white = {(r, c): grid[r][c] for r in range(n) for c in range(n)
             if grid[r][c] != "#" and grid[r][c].isalpha()}
    if not white:
        res["reason"] = "empty grid"; return res
    W = set(white)

    def runs(dr, dc):
        o = []
        for (r, c) in W:
            if (r - dr, c - dc) in W:
                continue
            w, rr, cc = "", r, c
            while (rr, cc) in W:
                w += white[(rr, cc)]; rr, cc = rr + dr, cc + dc
            o.append(w)
        return o

    hr, vr = runs(0, 1), runs(1, 0)
    allr = hr + vr
    bad_short = [w for w in allr if len(w) < 3]
    entries = [w for w in allr if len(w) >= 3]
    nonword = [w for w in entries if w not in DICT]

    def connected(cells):
        seen = {next(iter(cells))}; st = list(seen)
        while st:
            r, c = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (r + dr, c + dc)
                if nb in cells and nb not in seen:
                    seen.add(nb); st.append(nb)
        return len(seen) == len(cells)

    conn = connected(W)

    # declared clues from the Across:/Down: sections
    decl_a, decl_d, mode = [], [], None
    for ln in lines:
        s = ln.strip()
        if s.startswith("Across"): mode = "a"; continue
        if s.startswith("Down"): mode = "d"; continue
        m = re.match(r"\d+\.\s*([A-Za-z]+)$", s)
        if m:
            (decl_a if mode == "a" else decl_d).append(m.group(1).upper()) if mode else None
    actual_a = sorted(w for w in hr if len(w) >= 3)
    actual_d = sorted(w for w in vr if len(w) >= 3)
    mismatch = (sorted(decl_a) != actual_a) or (sorted(decl_d) != actual_d)

    valid = (not bad_short) and (not nonword) and conn and (not mismatch)
    res["valid"] = bool(valid)
    if not valid:
        rs = []
        if bad_short: rs.append(f"{len(bad_short)} short run(s)")
        if nonword: rs.append(f"{len(nonword)} nonword e.g. {nonword[:3]}")
        if not conn: rs.append("disconnected")
        if mismatch: rs.append("printed clues != grid runs")
        res["reason"] = "; ".join(rs) or "invalid"
    return res


def _size_from_name(name):
    m = re.search(r"_s(\d+)\.py$", name)
    return int(m.group(1)) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    progs = []
    for p in sorted(glob.glob(os.path.join(a.directory, "**", "*.py"), recursive=True)):
        sz = _size_from_name(os.path.basename(p))
        if sz is not None:
            progs.append((p, sz))
    if not progs:
        sys.exit(f"no *_sNN.py programs under {a.directory}")
    DICT = load_dict()
    tasks = [(p, sz, r) for (p, sz) in progs for r in range(a.repeats)]
    print(f"{len(progs)} programs x {a.repeats} runs = {len(tasks)} executions "
          f"(workers={a.workers}, timeout={a.timeout:.0f}s)", flush=True)

    def do(task):
        path, sz, _ = task
        out, err = run_script(path, a.timeout)
        if err:
            return {"prog": os.path.basename(path), "size": sz, "ran": 0, "valid": False, "reason": err}
        chk = validate_printed(out, sz, DICT)
        return {"prog": os.path.basename(path), "size": sz, "ran": 1,
                "valid": chk["valid"], "reason": chk["reason"]}

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(do, tasks):
            rows.append(rec); done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} ({sum(r['valid'] for r in rows)} valid)", flush=True)

    sizes = sorted({r["size"] for r in rows})
    def agg(rs):
        n = len(rs) or 1
        return {"n": len(rs), "ran": sum(r["ran"] for r in rs) / n, "valid": sum(r["valid"] for r in rs) / n}
    print(f"\n{'size':>5}{'runs':>7}{'ran%':>7}{'valid%':>8}")
    for s in sizes:
        aa = agg([r for r in rows if r["size"] == s])
        print(f"{s:>5}{aa['n']:>7}{aa['ran']*100:>6.0f}{aa['valid']*100:>8.1f}")
    ov = agg(rows)
    print(f"{'ALL':>5}{ov['n']:>7}{ov['ran']*100:>6.0f}{ov['valid']*100:>8.1f}")

    reasons = collections.Counter(r["reason"] for r in rows if not r["valid"])
    if reasons:
        print("\nfailure reasons:")
        for reason, c in reasons.most_common():
            print(f"  {c:>4}  {reason}")

    out = a.out or os.path.join(a.directory, "scriptrun_result.json")
    json.dump({"n_progs": len(progs), "repeats": a.repeats, "n_runs": len(rows),
               "overall": agg(rows), "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in sizes},
               "reasons": dict(reasons), "rows": rows},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
