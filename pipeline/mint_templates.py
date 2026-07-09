"""Mint a library of fillable NxN templates by offline rejection sampling.

For sizes with no real-grid corpus (11x11: NYT dailies are 15x15, Sundays 21x21),
generate candidate symmetric black-square patterns with the engine's own
constructor and KEEP only the ones the clean educational palette can actually
fill. Slow offline (minutes, ~9% of random 11x11 grids fill) so the emitted
GENERATION scripts stay fast: they just select-and-fill from the baked-in library.

Output matches build_template_library (data/templates_<size>.json), so
emit_template_generator consumes it unchanged.

    python pipeline/mint_templates.py --size 11 --target 80 --workers 5 --out data/templates_11.json
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
                  min_word_len=3, time_budget_s=float(size * 2)),
    )


def _mint_one(seed, fill_seeds, deadline):
    """Construct a random symmetric structure (seed), try to fill it. Keep if valid."""
    eng, size, full = _W["eng"], _W["size"], _W["full"]
    white = eng._make_structure(size, random.Random(seed))
    black = full - white
    if not black:                       # constructor fell back to a full white grid
        return None
    slots, cts = eng._slots_and_crossings(white, size)
    for fs in range(fill_seeds):
        rng = random.Random(1000 + seed * 7 + fs)
        t = time.perf_counter()
        a = eng._fill(slots, cts, _W["idx"], rng, _W["theme_set"], budget=300000,
                      deadline=time.perf_counter() + deadline)
        dt = time.perf_counter() - t
        if a and len(a) == len(slots):
            lay = eng._build_layout(white, size, slots, a)
            m = score(lay, _W["spec"], _W["words"], scores=_W["scores"],
                      runtime_s=dt, vocab_set=_W["clean_set"])
            if m["valid"] == 1:
                return {"black": sorted([list(b) for b in black]), "size": size,
                        "n_slots": len(slots), "fill_seed": fs,
                        "coverage": m["coverage"], "filler": m["filler_fraction"],
                        "seconds": round(dt, 2)}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--target", type=int, default=80, help="how many fillable templates to keep")
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--max-seconds", type=float, default=1500.0)
    ap.add_argument("--fill-seeds", type=int, default=2, help="fill attempts per candidate")
    ap.add_argument("--deadline", type=float, default=5.0, help="per fill-attempt seconds")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--engine", default=os.path.join(_ROOT, "generations", "gen3", "ac3_lcv.py"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(_ROOT, "data", f"templates_{args.size}.json")

    kept, seen = [], set()
    tried = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.engine, args.size)) as ex:
        futs = {ex.submit(_mint_one, s, args.fill_seeds, args.deadline): s
                for s in range(args.max_candidates)}
        for fut in as_completed(futs):
            tried += 1
            res = fut.result()
            if res is not None:
                key = tuple(map(tuple, res["black"]))
                if key not in seen:
                    seen.add(key)
                    kept.append(res)
                    print(f"  kept {len(kept):3d}/{args.target}  (tried {tried})  "
                          f"slots={res['n_slots']} cov={res['coverage']:.2f} "
                          f"filler={res['filler']:.2f} {res['seconds']}s", flush=True)
            if len(kept) >= args.target or (time.time() - t0) > args.max_seconds:
                break
        ex.shutdown(cancel_futures=True)

    result = {
        "size": args.size,
        "source": f"minted offline (random symmetric construction + {os.path.basename(args.engine)} fill)",
        "palette": f"build_clean_education_source(max_len={args.size})",
        "n_tried": tried,
        "n_fillable": len(kept),
        "fill_rate": round(len(kept) / tried, 4) if tried else 0.0,
        "engine": os.path.basename(args.engine),
        "templates": kept,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    print(f"\nminted {len(kept)} fillable {args.size}x{args.size} templates from {tried} "
          f"candidates ({100*result['fill_rate']:.0f}% fill) in {time.time()-t0:.0f}s -> {out}")
    return result


if __name__ == "__main__":
    main()
