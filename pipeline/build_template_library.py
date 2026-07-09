"""Distill the scraped NYT 15x15 grids into a library of KNOWN-FILLABLE templates.

The scrape (data/scraped/nyt_vocab.jsonl) gives 155 distinct real NYT black-square
patterns. Not all can be filled by our clean educational palette (dense 3-letter
interlock is the bottleneck). This script keeps only the black-square GEOMETRY of
each grid (never NYT's words), tries to fill it with the ac3_lcv engine over a few
seeds, and writes the templates that fill within budget to data/templates_15.json.

Those become the baked-in constant for the fixed-template generator family: a
program that inlines these patterns and fills them is guaranteed a fast, valid,
zero-filler crossword -- no random construction gamble.

    python pipeline/build_template_library.py --seeds 3 --deadline 8 --out data/templates_15.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score
from pipeline.word_source import build_clean_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-worker globals (built once in _init_worker to avoid re-serializing the palette)
_W: dict = {}


def _load_engine(path):
    spec = importlib.util.spec_from_file_location("engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _init_worker(engine_path, size):
    eng = _load_engine(engine_path)
    edu = build_clean_education_source(max_len=size)
    theme, fill = edu["targets"], edu["fill_words"]
    words = theme + fill
    _W.update(
        eng=eng, size=size, theme_set=set(theme), words=words,
        idx=eng._index_by_length(words), scores=edu["scores"], clean_set=edu["clean_set"],
        full={(r, c) for r in range(size) for c in range(size)},
        spec=Spec(size=size, topic_words=tuple(theme), require_symmetry=False,
                  min_word_len=3, time_budget_s=30.0),
    )


def _eval_template(tpl, seeds, deadline):
    eng, size = _W["eng"], _W["size"]
    white = _W["full"] - {tuple(b) for b in tpl["black"]}
    slots, cts = eng._slots_and_crossings(white, size)
    for s in range(seeds):
        rng = random.Random(s)
        t = time.perf_counter()
        a = eng._fill(slots, cts, _W["idx"], rng, _W["theme_set"], budget=300000,
                      deadline=time.perf_counter() + deadline)
        dt = time.perf_counter() - t
        if a and len(a) == len(slots):
            lay = eng._build_layout(white, size, slots, a)
            m = score(lay, _W["spec"], _W["words"], scores=_W["scores"],
                      runtime_s=dt, vocab_set=_W["clean_set"])
            if m["valid"] == 1:
                return {"black": tpl["black"], "size": size, "n_slots": len(slots),
                        "fill_seed": s, "coverage": m["coverage"],
                        "filler": m["filler_fraction"], "seconds": round(dt, 2)}
    return None


def distinct_templates(scrape_path):
    seen, out = set(), []
    for line in open(scrape_path, encoding="utf-8"):
        if not line.strip():
            continue
        rc = json.loads(line)["resulting crossword"]
        key = tuple(map(tuple, rc["black"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"black": [list(b) for b in rc["black"]], "size": rc["size"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", default=os.path.join(_ROOT, "data", "scraped", "nyt_vocab.jsonl"))
    ap.add_argument("--engine", default=os.path.join(_ROOT, "generations", "gen3", "ac3_lcv.py"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "templates_15.json"))
    ap.add_argument("--seeds", type=int, default=3, help="fill attempts per template")
    ap.add_argument("--deadline", type=float, default=8.0, help="per-attempt seconds")
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    templates = distinct_templates(args.scrape)
    kept = []
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.engine, args.size)) as ex:
        futs = {ex.submit(_eval_template, tpl, args.seeds, args.deadline): i
                for i, tpl in enumerate(templates)}
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res is not None:
                kept.append(res)
                print(f"  [{done:3d}/{len(templates)}] FILL slots={res['n_slots']:3d} "
                      f"seed={res['fill_seed']} cov={res['coverage']:.2f} "
                      f"filler={res['filler']:.2f} {res['seconds']}s", flush=True)
            else:
                print(f"  [{done:3d}/{len(templates)}] ----", flush=True)

    out = {
        "size": args.size,
        "source": "doshea/nyt_crosswords (black-square geometry only)",
        "palette": f"build_clean_education_source(max_len={args.size})",
        "n_candidates": len(templates),
        "n_fillable": len(kept),
        "engine": os.path.basename(args.engine),
        "templates": kept,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"\nkept {len(kept)}/{len(templates)} fillable templates "
          f"({100*len(kept)/len(templates):.0f}%) in {time.time()-t_start:.0f}s -> {args.out}")
    return out


if __name__ == "__main__":
    main()
