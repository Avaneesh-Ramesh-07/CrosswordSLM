"""Gauge self-contained tuned-SLM crossword programs (exercise the hardcoding).

Runs each emitted `generate_crossword` program with **word_source=None** so it MUST fill from
its own baked `_WORDS`, then validates the crossword it returns with a standalone checker
(structure + every entry a real dictionary word). No palette is injected and the scoring
harness (`score_one`/`harness.scorer`) is NOT used -- this simply runs the model's output and
judges the crossword on its own terms, the same way the 36 hardcoded dataset programs were
gauged (36/36 valid).

Each program runs in its own child process with a hard timeout, so a hung or crashing sample
can't stall or take down the gauge.

    python pipeline/gauge_selfcontained.py runs/eval/slm_gen        # dir with progs/*.py [+ specs.jsonl]
    python pipeline/gauge_selfcontained.py <dir> --timeout 60 --out <file.json>

Size per program is read from `specs.jsonl` if present, else parsed from the `_sNN` filename
suffix (e.g. prog_007_s11.py -> size 11).
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DICT_PATH = os.path.join(_ROOT, "data", "wordlists", "words_alpha.txt")

# Per-size execution budget (seconds) — the project's contract budgets; used only to record
# runtime_ok, NOT as the process kill timeout (that is --timeout, deliberately generous).
BUDGET = {7: 3, 9: 5, 11: 12, 15: 30}


def load_dict(path=_DICT_PATH) -> set:
    with open(path, encoding="utf-8") as fh:
        return {w.strip().upper() for w in fh if w.strip()}


def check_crossword(layout, size, DICT, require_symmetry=False, min_len=3) -> dict:
    """Standalone validity check — mirrors harness.scorer's `valid` rules, but judges words
    against a real dictionary (`DICT`) instead of a supplied palette."""
    res = {"valid": False, "n_entries": 0, "crossings": 0, "density": 0.0,
           "dict_frac": 0.0, "reason": ""}
    if not isinstance(layout, dict) or "across" not in layout or "down" not in layout:
        res["reason"] = "bad schema"
        return res
    try:
        across = list(layout["across"]); down = list(layout["down"])
    except TypeError:
        res["reason"] = "across/down not iterable"
        return res
    dims_ok = layout.get("rows") == size and layout.get("cols") == size
    N = lambda w: str(w).strip().upper()
    grid = {}; conflict = False; oob = False

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
            place(N(e["answer"]), int(e["row"]), int(e["col"]), 0, 1)
        for e in down:
            place(N(e["answer"]), int(e["row"]), int(e["col"]), 1, 0)
    except (KeyError, TypeError, ValueError):
        res["reason"] = "entry missing answer/row/col"
        return res

    white = set(grid)
    if not white:
        res["reason"] = "empty grid"
        return res

    def runs(dr, dc):
        out = []
        for (r, c) in white:
            if (r - dr, c - dc) in white:
                continue
            w, rr, cc, L = "", r, c, 0
            while (rr, cc) in white:
                w += grid[(rr, cc)]; L += 1; rr, cc = rr + dr, cc + dc
            out.append((r, c, w, L))
        return out

    hruns, vruns = runs(0, 1), runs(1, 0)
    allruns = hruns + vruns
    bad_short = [x for x in allruns if x[3] < min_len]
    actual_a = {(r, c, w) for (r, c, w, l) in hruns}
    actual_d = {(r, c, w) for (r, c, w, l) in vruns}
    claimed_a = {(int(e["row"]), int(e["col"]), N(e["answer"])) for e in across}
    claimed_d = {(int(e["row"]), int(e["col"]), N(e["answer"])) for e in down}
    entries = [w for (r, c, w, l) in allruns if l >= min_len]
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

    conn = connected(white)
    sym = True
    if require_symmetry:
        sym = all(((r, c) in white) == ((size - 1 - r, size - 1 - c) in white)
                  for r in range(size) for c in range(size))

    across_cells = {(r, c + i) for (r, c, w, l) in hruns if l >= min_len for i in range(l)}
    down_cells = {(r + i, c) for (r, c, w, l) in vruns if l >= min_len for i in range(l)}
    res["crossings"] = len(across_cells & down_cells)
    res["n_entries"] = len(across) + len(down)
    res["density"] = round(len(white) / (size * size), 4)
    res["dict_frac"] = round(sum(1 for w in entries if w in DICT) / len(entries), 4) if entries else 0.0

    valid = (not conflict and not oob and not bad_short
             and actual_a == claimed_a and actual_d == claimed_d
             and not nonword and conn and (sym or not require_symmetry) and dims_ok)
    res["valid"] = bool(valid)
    if not valid:
        rs = []
        if conflict: rs.append("conflict")
        if oob: rs.append("oob")
        if bad_short: rs.append(f"{len(bad_short)} short")
        if actual_a != claimed_a: rs.append("across-mismatch")
        if actual_d != claimed_d: rs.append("down-mismatch")
        if nonword: rs.append(f"{len(nonword)} nonword {nonword[:3]}")
        if not conn: rs.append("disconnected")
        if not dims_ok: rs.append("dims")
        res["reason"] = "; ".join(rs)
    return res


def _worker(code, size, q):
    """Child process: exec the program and call generate_crossword with NO word_source."""
    try:
        # __name__ != "__main__" so a standalone `if __name__ == "__main__"` block does NOT
        # run here (we call generate_crossword ourselves); set it explicitly for robustness.
        ns = {"__name__": "__gauge_prog__"}
        exec(compile(code, "<prog>", "exec"), ns)
        fn = ns.get("generate_crossword")
        if fn is None:
            q.put({"error": "no generate_crossword"}); return
        t = time.time()
        # self-contained: word_source=None -> program must use its own baked _WORDS.
        # Report the REAL error (a program with no _WORDS raises 'NoneType' not iterable
        # here); do NOT fall back to a 2-arg call, which masks that as "missing size".
        lay = fn("vocabulary", None, size)
        q.put({"layout": lay, "runtime_s": round(time.time() - t, 4)})
    except Exception as e:
        q.put({"error": f"{type(e).__name__}: {e}"})


def run_one(code, size, timeout):
    """Run one program in an isolated child process; return (result_dict_or_error, runtime)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, size, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"error": "timeout"}, timeout
    try:
        out = q.get_nowait()
    except Exception:
        return {"error": "no output (crashed)"}, 0.0
    return out, out.get("runtime_s", 0.0)


def _size_from_name(name):
    m = re.search(r"_s(\d+)\.py$", name)
    return int(m.group(1)) if m else None


def discover(directory):
    """Yield (prog_path, size) using specs.jsonl if present, else the _sNN filename suffix."""
    specs = os.path.join(directory, "specs.jsonl")
    if os.path.exists(specs):
        for line in open(specs, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield os.path.join(directory, rec["prog_file"]), int(rec["size"])
        return
    for p in sorted(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)):
        if os.sep + "raw" + os.sep in p:
            continue
        sz = _size_from_name(os.path.basename(p))
        if sz is not None:
            yield p, sz


def agg(rows):
    n = len(rows) or 1
    v = [r for r in rows if r["valid"]]
    vn = len(v) or 1
    return {"n": len(rows),
            "ran": sum(r["ran"] for r in rows) / n,
            "valid": sum(r["valid"] for r in rows) / n,
            "dict": sum(r["dict_frac"] for r in rows) / n,
            "cross": sum(r["crossings"] for r in v) / vn,
            "ent": sum(r["n_entries"] for r in v) / vn,
            "dens": sum(r["density"] for r in v) / vn,
            "rt": sum(r["runtime_s"] for r in rows) / n}


def table(rows, sizes):
    hdr = (f"{'size':>5}{'n':>5}{'ran%':>7}{'valid%':>8}{'dict%':>7}"
           f"{'cross':>7}{'entries':>8}{'density':>8}{'rt':>7}")
    print(hdr); print("-" * len(hdr))
    for s in sizes:
        a = agg([r for r in rows if r["size"] == s])
        print(f"{s:>5}{a['n']:>5}{a['ran']*100:>6.0f}{a['valid']*100:>8.0f}{a['dict']*100:>7.0f}"
              f"{a['cross']:>7.0f}{a['ent']:>8.0f}{a['dens']:>8.2f}{a['rt']:>7.2f}")
    a = agg(rows); print("-" * len(hdr))
    print(f"{'ALL':>5}{a['n']:>5}{a['ran']*100:>6.0f}{a['valid']*100:>8.0f}{a['dict']*100:>7.0f}"
          f"{a['cross']:>7.0f}{a['ent']:>8.0f}{a['dens']:>8.2f}{a['rt']:>7.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="dir with progs/*.py (+ optional specs.jsonl)")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-program hard-kill seconds")
    ap.add_argument("--require-symmetry", action="store_true",
                    help="also require 180-deg symmetry (off by default, matching the harness eval)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    progs = list(discover(a.directory))
    if not progs:
        sys.exit(f"no programs found under {a.directory} (need progs/*.py or specs.jsonl)")
    print(f"gauging {len(progs)} programs from {a.directory} "
          f"(self-contained: word_source=None; timeout {a.timeout:.0f}s)\n", flush=True)
    DICT = load_dict()
    print(f"dictionary: {len(DICT):,} words\n", flush=True)

    rows = []
    for i, (path, size) in enumerate(progs):
        code = open(path, encoding="utf-8").read()
        rec = {"prog": os.path.basename(path), "size": size, "ran": 0,
               "valid": False, "dict_frac": 0.0, "crossings": 0, "n_entries": 0,
               "density": 0.0, "runtime_s": 0.0, "reason": ""}
        if not code.strip():
            rec["reason"] = "empty program (no code emitted)"
        else:
            out, rt = run_one(code, size, a.timeout)
            rec["runtime_s"] = rt
            if "error" in out:
                rec["reason"] = out["error"]
            else:
                rec["ran"] = 1
                chk = check_crossword(out["layout"], size, DICT, require_symmetry=a.require_symmetry)
                rec.update({k: chk[k] for k in ("valid", "dict_frac", "crossings", "n_entries", "density", "reason")})
        rows.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(progs):
            nv = sum(r["valid"] for r in rows)
            print(f"  {i+1}/{len(progs)} done ({nv} valid so far)", flush=True)

    sizes = sorted({r["size"] for r in rows})
    print()
    table(rows, sizes)

    # a few example failures per size, to eyeball what's going wrong
    print("\nsample failures:")
    for s in sizes:
        fails = [r for r in rows if r["size"] == s and not r["valid"]][:3]
        for r in fails:
            print(f"  [{s:>2}] {r['prog']}: {r['reason']}")

    out = a.out or os.path.join(a.directory, "gauge_result.json")
    summary = {"dir": a.directory, "n": len(rows), "timeout": a.timeout,
               "require_symmetry": a.require_symmetry,
               "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in sizes},
               "overall": agg(rows), "rows": rows}
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
