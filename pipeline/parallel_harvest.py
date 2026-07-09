"""Parallel in-process harvester for FAST large-scale dataset generation.

Evaluates many (program, spec) tasks concurrently across CPU cores
(ProcessPoolExecutor), each in-process (no sandbox subprocess, no word_source
re-serialization). The clean palette is built ONCE per worker. Trusted teacher
code only. Unions programs from one or more generation manifests, harvests
solutions + labeled negatives, and writes the dataset (train/dev/eval + negatives).

    python pipeline/parallel_harvest.py --out runs/bulk --n-specs 500 --n-draws 2 \
        --per-program-cap 250 --gen-dirs generations/gen1,generations/gen2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import build_dataset, elites, harvest
from pipeline.oe_evaluator import evaluate_code
from pipeline.spec_generator import SpecRecord, generate_specs, save_specs
from pipeline.word_source import build_clean_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# per-worker globals (built once per process in the initializer)
_WS = _SCORES = _VOCAB = None


def _init_worker():
    global _WS, _SCORES, _VOCAB
    edu = build_clean_education_source()
    _WS = {"theme": edu["targets"], "fill": edu["fill_words"]}
    _SCORES = edu["scores"]
    _VOCAB = edu["clean_set"]


def _eval_task(task):
    """task = (code, spec_dict, n_draws) -> harvest row. Runs in a worker process."""
    code, spec_dict, n_draws = task
    rec = SpecRecord(**spec_dict)
    spec = rec.to_scorer_spec(topic_words=_WS["theme"])
    out = evaluate_code(code, spec, _WS, scores=_SCORES, n_draws=n_draws,
                        vocab_set=_VOCAB, in_process=True)
    return {"spec_id": rec.spec_id, "code": code,
            "metrics": out["metrics"], "artifacts": out["artifacts"]}


def _collect_programs(gen_dirs, skip):
    """Union of manifest programs across gen_dirs (dedup by name); resolve code +
    allowed sizes. Resolution order: the gen dir itself, then every prior
    generations/gen* dir, then teachers/, seeds/ (so baseline programs named in a
    later manifest but living in an earlier dir still resolve)."""
    prior = sorted(glob.glob(os.path.join(_ROOT, "generations", "gen*")))
    search = prior + [os.path.join(_ROOT, d) for d in ("teachers", "seeds")]
    progs = {}
    for gd in gen_dirs:
        gdp = gd if os.path.isabs(gd) else os.path.join(_ROOT, gd)
        mpath = os.path.join(gdp, "manifest.json")
        if not os.path.exists(mpath):
            continue
        for p in json.load(open(mpath, encoding="utf-8")).get("programs", []):
            name = p["name"]
            if name in progs or name in skip:
                continue
            for base in [gdp] + search:
                fp = os.path.join(base, name + ".py")
                if os.path.exists(fp):
                    with open(fp, encoding="utf-8") as fh:
                        progs[name] = {"code": fh.read(), "sizes": set(p.get("sizes", []))}
                    break
    return progs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-dirs", default="generations/gen1,generations/gen2")
    ap.add_argument("--sizes", default="7,9,11")
    ap.add_argument("--n-specs", type=int, default=500)
    ap.add_argument("--n-draws", type=int, default=2)
    ap.add_argument("--per-program-cap", type=int, default=250)
    ap.add_argument("--per-negative-cap", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0, help="0 -> os.cpu_count()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="weak_greedy", help="programs to exclude (comma-sep)")
    args = ap.parse_args()

    workers = args.workers or (os.cpu_count() or 4)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    gen_dirs = [d.strip() for d in args.gen_dirs.split(",") if d.strip()]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    os.makedirs(args.out, exist_ok=True)
    specs = generate_specs(args.n_specs, seed=args.seed, sizes=sizes)
    save_specs(specs, os.path.join(args.out, "specs.jsonl"))
    specs_by_id = {s.spec_id: s for s in specs}

    progs = _collect_programs(gen_dirs, skip)
    tasks = []
    for info in progs.values():
        allowed = info["sizes"]
        for spec in specs:
            if allowed and spec.size not in allowed:
                continue   # size routing
            tasks.append((info["code"], spec.as_dict(), args.n_draws))
    print(f"programs={len(progs)} ({sorted(progs)})")
    print(f"specs={len(specs)} tasks={len(tasks)} workers={workers} n_draws={args.n_draws}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futs = [ex.submit(_eval_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 250 == 0 or i == len(tasks):
                nv = sum(1 for r in rows if r["metrics"].get("valid"))
                print(f"  {i}/{len(tasks)} evaluated | valid so far={nv}", flush=True)

    data_dir = os.path.join(args.out, "dataset")
    out = harvest.process_harvest(rows, specs_by_id,
                                  per_program_cap=args.per_program_cap,
                                  per_negative_cap=args.per_negative_cap)
    counts = build_dataset.build(out["solutions"], data_dir)
    neg = build_dataset.write_negatives(out["negatives"], data_dir)
    with open(os.path.join(args.out, "harvest.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    elites.update_elites(rows, os.path.join(_ROOT, "data", "elites"), gen="bulk")

    print(f"\nsolutions={out['n_solutions']} ({out['kind_counts']}) "
          f"negatives={out['n_negatives']} ({out['failure_counts']}) "
          f"distinct_programs={out['n_distinct_programs']}")
    print(f"dataset splits: {counts}  (DPO neg={neg['pool']}, eval neg={neg['eval']})  -> {data_dir}")
    total = sum(counts.values())
    print(f"TOTAL SFT records: {total}  |  + {neg['pool']} DPO negatives = {total + neg['pool']} records")


if __name__ == "__main__":
    main()
