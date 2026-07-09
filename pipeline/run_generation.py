"""Orchestrate ONE generation of the evolutionary teacher (local, GPU-free).

Runs the generation's programs through the strict verifier across specs (via
run_openevolve --dry-run, which also builds the dataset + persists negatives),
then produces the scorecard, updates the persistent elite library, and exports the
elite frontier for the next generation / the real OpenEvolve run on Colab.

    python pipeline/run_generation.py --gen-dir generations/gen1 --out runs/gen1 \
        --clean-palette --sizes 7,9,11 --n-specs 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import analyze_generation, elites
from pipeline.analyze_generation import _load_jsonl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _program_names(gen_dir):
    with open(os.path.join(gen_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    progs = manifest.get("programs", [])
    names = [p["name"] for p in progs]
    # per-program size routing (skip sizes a program can't fill) from optional "sizes"
    sizes = {p["name"]: p["sizes"] for p in progs if p.get("sizes")}
    return names, manifest.get("gen", 1), sizes


def run_generation(gen_dir, out_dir, sizes="7,9,11", n_specs=24, seed=0,
                   clean_palette=True, hint_weights_path="", extra_dirs="teachers,seeds",
                   elites_dir=None, next_seed_dir=None, top_k=8, n_draws=3, in_process=True):
    names, gen, prog_sizes = _program_names(gen_dir)
    elites_dir = elites_dir or os.path.join(_ROOT, "data", "elites")
    seeds_dir = ",".join([os.path.relpath(gen_dir, _ROOT)] + extra_dirs.split(","))
    seed_sizes = ";".join(f"{n}:{','.join(str(s) for s in szs)}" for n, szs in prog_sizes.items())

    cmd = [sys.executable, os.path.join(_ROOT, "pipeline", "run_openevolve.py"),
           "--dry-run", "--sizes", sizes, "--n-specs", str(n_specs), "--seed", str(seed),
           "--n-draws", str(n_draws),
           "--seeds", ",".join(names), "--seeds-dir", seeds_dir, "--out", out_dir]
    if in_process:
        cmd.append("--in-process")   # fast path: trusted teacher code, no subprocess
    if seed_sizes:
        cmd += ["--seed-sizes", seed_sizes]
    if clean_palette:
        cmd.append("--clean-palette")
    if hint_weights_path:
        cmd += ["--hint-weights", hint_weights_path]
    print("running:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

    # Learnings: per-program + per-heuristic scorecard.
    analyze_generation.scorecard(out_dir, gen_dir)

    # Elite library: carry the best valid programs forward.
    harvest_rows = _load_jsonl(os.path.join(out_dir, "harvest.jsonl"))
    stats = elites.update_elites(harvest_rows, elites_dir, gen)
    print(f"\nelites: {stats['n_cells']} cells (+{stats['n_added']} this gen) -> {elites_dir}")
    if next_seed_dir:
        exported = elites.export_seeds(elites_dir, next_seed_dir, top_k=top_k)
        print(f"exported {len(exported)} elite seeds -> {next_seed_dir}: {exported}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="7,9,11")
    ap.add_argument("--n-specs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-palette", action="store_true")
    ap.add_argument("--hint-weights", default="", help="prior scorecard.json to bias hints")
    ap.add_argument("--next-seed-dir", default="", help="dir to export elite seeds into")
    ap.add_argument("--n-draws", type=int, default=3,
                    help="draws per candidate; solution must be valid on ALL (reliability filter)")
    ap.add_argument("--extra-dirs", default="teachers,seeds",
                    help="extra dirs (relative to root) to resolve manifest program names from")
    args = ap.parse_args()
    run_generation(args.gen_dir, args.out, sizes=args.sizes, n_specs=args.n_specs,
                   seed=args.seed, clean_palette=args.clean_palette,
                   hint_weights_path=args.hint_weights, next_seed_dir=args.next_seed_dir or None,
                   n_draws=args.n_draws, extra_dirs=args.extra_dirs)


if __name__ == "__main__":
    main()
