"""Harvest real NYT-style crosswords and format them as {spec, resulting crossword}.

We want REAL, human-built, "grid-like" American crosswords (square grids with
minimal black squares, fully interlocked) whose fill overlaps our educational
vocabulary -- the intersection of crossword-worthy words and SAT words
(`word_source.build_education_source()['targets']`, ~2.7k words). These become
supervised {SPEC -> crossword} pairs whose SPEC is modeled on the LLM-evo task
spec (`spec_generator.render_spec`).

SOURCE: the public `doshea/nyt_crosswords` corpus (NYT dailies+Sundays 1976-2017,
one JSON per puzzle: flat row-major `grid` with "." for black, `answers.across`
/`answers.down`, `size.{rows,cols}`). We do NOT scrape nytimes.com (paywalled,
ToS-forbidden); this is the canonical open dump of the same grids. Clone it:
    git clone --depth 1 https://github.com/doshea/nyt_crosswords <corpus_dir>

FILTERS (this is what "NYT-like, square, minimal black, not free-form" means):
  * SQUARE only              -- size.rows == size.cols and grid is that square.
  * clean single-letter grid -- every non-black cell is one A-Z letter (drops
                                rebus / diagramless / variety = the "free-form").
  * minimal black            -- black fraction <= --max-black-frac (NYT ~0.13-0.17).
  * fully interlocked, min-3 -- every maximal white run (across AND down) is >= 3,
                                so every white cell is checked both ways (NYT-legal).

Each kept puzzle is scored by vocab_fraction = (#entries in the vocab set) /
(#entries). We emit only puzzles with vocab_fraction >= --min-vocab-frac.

The SPEC is guaranteed to MATCH its crossword: stated size == grid size; stated
symmetry == actual 180-degree symmetry; stated min length (3) <= actual min run;
stated white-density floor <= actual white fraction; stated "at least Y%
vocabulary" uses Y = floor(vocab_fraction*100), so the actual fraction satisfies it.

Usage (Git Bash / WSL, stdlib-only, Python 3.11+):
    python pipeline/scrape_crosswords.py --corpus <dir> --analyze
    python pipeline/scrape_crosswords.py --corpus <dir> --min-vocab-frac 0.15 --out data/scraped/nyt_vocab.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os

if __package__ in (None, ""):
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import MIN_WORD_LEN
from pipeline.word_source import build_clean_education_source, build_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLACK = "."


def build_vocab(kind, max_len=21):
    """Return (vocab_set, label). 'strict' = SAT n crossword-optimal (~2.7k, the
    literal intersection). 'clean' = the project's educational palette (~27k:
    real dictionary words that are frequency-gettable-or-SAT and crossword-worthy)."""
    if kind == "clean":
        edu = build_clean_education_source(max_len=max_len)
        return set(edu["allowed"]), f"clean_educational_palette({edu['n_allowed']})"
    edu = build_education_source(max_len=max_len)
    return set(edu["targets"]), f"sat_x_crossword_optimal({edu['n_vocab']})"

_SIGNATURE_BLOCK = (
    "Write a single self-contained Python function:\n"
    "    generate_crossword(topic: str, word_source: list[str], size: int) -> dict\n"
    "It must construct and fill the grid and return a layout dict of the form:\n"
    '    {"rows": int, "cols": int,\n'
    '     "cells": [{"r": int, "c": int, "letter": str, "number": int (optional)}],\n'
    '     "across": [{"number", "row", "col", "answer", "len"}], "down": [ ...same... ]}'
)


# --- grid geometry ----------------------------------------------------------

def _to_2d(flat, rows, cols):
    """Flat row-major list -> 2D grid, or None if it isn't a clean single-letter grid."""
    if len(flat) != rows * cols:
        return None
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = flat[r * cols + c]
            if cell is None:
                return None
            cell = str(cell).strip().upper()
            if cell == BLACK or cell == "":
                row.append(BLACK)
            elif len(cell) == 1 and cell.isalpha():
                row.append(cell)
            else:
                # multi-char (rebus) or non-alpha -> not a clean square grid
                return None
        grid.append(row)
    return grid


def _white(grid, size):
    return {(r, c) for r in range(size) for c in range(size) if grid[r][c] != BLACK}


def _symmetric(white, size):
    """180-degree rotational symmetry of the black pattern (== scorer's definition)."""
    return all(((r, c) in white) == ((size - 1 - r, size - 1 - c) in white)
               for r in range(size) for c in range(size))


def _slots(grid, size, horizontal):
    """Yield (number-start (r,c), [cells]) for every maximal white run (any length)."""
    out = []
    for line in range(size):
        run = []
        for idx in range(size):
            r, c = (line, idx) if horizontal else (idx, line)
            if grid[r][c] != BLACK:
                run.append((r, c))
            else:
                if run:
                    out.append(run)
                run = []
        if run:
            out.append(run)
    return out


def _entries(grid, size):
    """Return (across, down, min_run_len). Each entry: dict(number,row,col,answer,len).

    Numbering follows standard American rules: a cell is numbered if it starts an
    across run and/or a down run; numbers increase left-to-right, top-to-bottom.
    The answer is READ FROM THE GRID (grid is source of truth), never from the
    corpus's answer list. min_run_len is the shortest maximal run in either
    direction -- < 3 means the grid is not NYT-legal (unchecked or 2-letter).
    """
    across_runs = _slots(grid, size, horizontal=True)
    down_runs = _slots(grid, size, horizontal=False)
    min_run = min((len(run) for run in across_runs + down_runs), default=0)

    starts = {run[0] for run in across_runs} | {run[0] for run in down_runs}
    number = {}
    n = 0
    for r in range(size):
        for c in range(size):
            if (r, c) in starts:
                n += 1
                number[(r, c)] = n

    def build(runs):
        items = []
        for run in runs:
            r, c = run[0]
            items.append({
                "number": number[(r, c)],
                "row": r, "col": c,
                "answer": "".join(grid[rr][cc] for rr, cc in run),
                "len": len(run),
            })
        items.sort(key=lambda e: e["number"])
        return items

    return build(across_runs), build(down_runs), min_run


# --- spec rendering (modeled on spec_generator.render_spec) ------------------

def render_spec(size, vocab_pct, require_symmetry, density_floor, min_word_len=MIN_WORD_LEN):
    """Natural-language SPEC for a scraped puzzle, faithful to the evo-task spec.

    Leads with the user's target phrasing ("... at least Y% of the words being
    vocabulary") and keeps render_spec's signature block + hard rules. Every
    stated constraint is TRUE of the resulting crossword (see module docstring)."""
    rules = [
        f"the grid is exactly {size} x {size}",
        f"every white run (across and down) is at least {min_word_len} letters, and "
        "every white cell is checked (part of both an across and a down entry)",
        "every entry is a word drawn from word_source, never invented or hardcoded",
        f"at least {vocab_pct}% of the entries are vocabulary words (from the topic vocabulary)",
        "all white cells form a single connected region",
    ]
    if require_symmetry:
        rules.append("the black squares are placed with 180-degree rotational symmetry")
    rules_text = "".join(f"\n  - {r};" for r in rules)
    return (
        f'Create me a {size} by {size} crossword with at least {vocab_pct}% of the words '
        f'being vocabulary. Generate a {size}x{size} fixed-grid, American-style crossword '
        f'on the topic "SAT vocabulary".\n'
        f"{_SIGNATURE_BLOCK}\n"
        f"Hard rules:{rules_text}\n"
        f"Aim for a white-square density of at least {density_floor:.2f}.\n"
        f"Output only the Python code."
    )


# --- corpus walk ------------------------------------------------------------

def iter_puzzles(corpus_dir):
    for dirpath, _dirs, files in os.walk(corpus_dir):
        if os.sep + ".git" in dirpath:
            continue
        for name in files:
            if name.endswith(".json"):
                path = os.path.join(dirpath, name)
                try:
                    with open(path, encoding="utf-8") as fh:
                        yield path, json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue


def process(raw, vocab, vocab_label=""):
    """Raw puzzle dict -> a result dict, or (None, reason) if it's filtered out."""
    size_obj = raw.get("size") or {}
    rows, cols = size_obj.get("rows"), size_obj.get("cols")
    if not (isinstance(rows, int) and isinstance(cols, int)):
        return None, "no_size"
    if rows != cols:
        return None, "not_square"
    size = rows
    if size < 5:
        return None, "too_small"

    grid = _to_2d(raw.get("grid") or [], rows, cols)
    if grid is None:
        return None, "rebus_or_dirty_grid"

    white = _white(grid, size)
    n_cells = size * size
    black_frac = 1.0 - len(white) / n_cells

    across, down, min_run = _entries(grid, size)
    if min_run < MIN_WORD_LEN:
        return None, "unchecked_or_short_run"

    words = [e["answer"] for e in across] + [e["answer"] for e in down]
    if not words:
        return None, "no_entries"
    # vocab_fraction counts ENTRIES that are vocabulary words (with multiplicity),
    # so it is exactly "% of the entries that are vocabulary". vocab_words is the
    # de-duplicated list for inspection.
    n_vocab = sum(1 for w in words if w in vocab)
    vocab_words = sorted({w for w in words if w in vocab})
    vocab_fraction = n_vocab / len(words)

    black = sorted([r, c] for r in range(size) for c in range(size) if grid[r][c] == BLACK)
    result = {
        "size": size,
        "grid": grid,
        "black": black,
        "n_black": len(black),
        "white_fraction": round(len(white) / n_cells, 4),
        "black_fraction": round(black_frac, 4),
        "symmetric": _symmetric(white, size),
        "min_word_len": min_run,
        "across": across,
        "down": down,
        "words": words,
        "n_words": len(words),
        "vocab_words": vocab_words,
        "n_vocab_words": n_vocab,
        "vocab_fraction": round(vocab_fraction, 4),
        "vocab_set": vocab_label,
        "source": {
            "corpus": "doshea/nyt_crosswords",
            "publisher": raw.get("publisher") or "The New York Times",
            "date": raw.get("date"),
            "dow": raw.get("dow"),
            "author": raw.get("author"),
            "title": raw.get("title"),
        },
    }
    return result, "kept_pre_vocab"


def make_record(result):
    """Wrap a processed puzzle into the {spec, resulting crossword} training pair.

    Y and the density floor are floored from EXACT integer counts (not the rounded
    display fields) so 'at least Y%' / 'at least D density' can never be rounded up
    into a claim the grid doesn't satisfy."""
    n_cells = result["size"] * result["size"]
    n_white = n_cells - result["n_black"]
    vocab_pct = result["n_vocab_words"] * 100 // result["n_words"]       # floor -> always true
    density_floor = math.floor(n_white * 100 / n_cells) / 100           # floor -> always true
    spec = render_spec(
        size=result["size"],
        vocab_pct=vocab_pct,
        require_symmetry=result["symmetric"],
        density_floor=density_floor,
    )
    return {"spec": spec, "resulting crossword": result}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="path to a cloned doshea/nyt_crosswords tree")
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "scraped", "nyt_vocab.jsonl"))
    ap.add_argument("--vocab-set", choices=("strict", "clean"), default="clean",
                    help="'strict'=SAT n crossword-optimal (~2.7k); 'clean'=educational palette (~27k)")
    ap.add_argument("--max-black-frac", type=float, default=0.18, help="'minimal black spaces' ceiling")
    ap.add_argument("--top-n", type=int, default=160,
                    help="keep the N most vocab-dense grids (0 = use --min-vocab-frac instead)")
    ap.add_argument("--min-vocab-frac", type=float, default=0.0,
                    help="if --top-n 0: keep puzzles with >= this vocab fraction")
    ap.add_argument("--analyze", action="store_true", help="print distributions only; write nothing")
    args = ap.parse_args()

    vocab, label = build_vocab(args.vocab_set)
    print(f"vocab set [{args.vocab_set}]: {len(vocab):,} words -> {label}")

    reasons = {}
    vocab_hist = {}                          # vocab-% bucket (of square+minimal-black+legal) -> count
    struct = []                              # every grid passing square+minimal-black+legal filters
    total = 0
    for _path, raw in iter_puzzles(args.corpus):
        total += 1
        result, reason = process(raw, vocab, label)
        if result is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        if result["black_fraction"] > args.max_black_frac:
            reasons["too_much_black"] = reasons.get("too_much_black", 0) + 1
            continue
        reasons["structural_ok"] = reasons.get("structural_ok", 0) + 1
        bucket = int(result["vocab_fraction"] * 100) // 10 * 10
        vocab_hist[bucket] = vocab_hist.get(bucket, 0) + 1
        struct.append(result)

    struct.sort(key=lambda r: (-r["vocab_fraction"], r["size"]))
    if args.top_n:
        kept = struct[:args.top_n]
        cutoff = kept[-1]["vocab_fraction"] if kept else 0.0
        selector = f"top {args.top_n} most vocab-dense (cutoff vocab_fraction >= {cutoff:.1%})"
    else:
        kept = [r for r in struct if r["vocab_fraction"] >= args.min_vocab_frac]
        selector = f">= {args.min_vocab_frac:.0%} vocab"

    print(f"\nscanned {total:,} puzzles")
    print("filter tally:")
    for k in sorted(reasons, key=lambda k: -reasons[k]):
        print(f"  {k:26s} {reasons[k]:>7,}")
    struct_ok = reasons.get("structural_ok", 0)
    print(f"\nvocab-fraction histogram over the {struct_ok:,} square+minimal-black+legal grids:")
    for b in sorted(vocab_hist):
        bar = "#" * min(60, vocab_hist[b] * 60 // max(vocab_hist.values()))
        print(f"  {b:3d}-{b + 9:3d}% : {vocab_hist[b]:>6,} {bar}")

    sizes = {}
    for r in kept:
        sizes[r["size"]] = sizes.get(r["size"], 0) + 1
    print(f"\nselected: {selector} -> {len(kept):,} puzzles")
    print("  kept by size:", dict(sorted(sizes.items())))
    if kept:
        print("  most vocab-dense examples:")
        for r in kept[:5]:
            print(f"    {r['size']}x{r['size']}  {r['vocab_fraction']:.0%}  "
                  f"{r['n_vocab_words']}/{r['n_words']}  {r['source']['date']}  "
                  f"e.g. {r['vocab_words'][:6]}")

    if args.analyze:
        print("\n[analyze] no output written.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for result in kept:
            fh.write(json.dumps(make_record(result), ensure_ascii=False) + "\n")
    print(f"\nwrote {len(kept):,} records -> {args.out}")


if __name__ == "__main__":
    main()
