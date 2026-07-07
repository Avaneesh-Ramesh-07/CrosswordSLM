"""Driver: run OpenEvolve across specs to harvest (spec -> program) training data.

Real runs happen on Colab (needs `openevolve` installed + an OpenAI-compatible
LLM endpoint, e.g. Qwen served by vLLM). For each spec the driver:
  1. writes a problem-context JSON (word_source, scores, spec fields),
  2. points our evaluator (pipeline.oe_evaluator.evaluate) at it via env vars,
  3. runs OpenEvolve seeded with the reference generator (EVOLVE-BLOCK wrapped),
  4. every evaluated candidate is appended to a harvest JSONL by the evaluator.
After all specs it processes the harvest into train/dev/test chat JSONL.

`--dry-run` skips OpenEvolve entirely and instead scores the seed program once
per spec, so the full orchestration + harvest + dataset build is testable with no
GPU/LLM. Use `--include-common-fill` in a dry run so the seed can actually fill
and you can see non-empty solutions flow through.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import build_dataset, harvest, oe_evaluator
from pipeline.spec_generator import generate_specs, save_specs
from pipeline.word_source import build_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SEEDS_DIR = os.path.join(_ROOT, "seeds")
_EVALUATOR = os.path.join(_ROOT, "pipeline", "oe_evaluator.py")


def make_initial_program(seed_path, out_path):
    """Copy a seed generator and wrap its body in EVOLVE-BLOCK markers so
    OpenEvolve evolves the algorithm (imports stay outside the block)."""
    src = open(seed_path, encoding="utf-8").read()
    marker_in, marker_out = "\n# EVOLVE-BLOCK-START\n", "\n# EVOLVE-BLOCK-END\n"
    if "import random" in src:
        src = src.replace("import random\n", "import random\n" + marker_in, 1) + marker_out
    else:
        src = marker_in + src + marker_out
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return out_path


def write_config(out_path, model, api_base, iterations):
    """Write an OpenEvolve config.yaml. NOTE: confirm keys against the installed
    openevolve version on Colab; adjust if v0.3.x renamed anything."""
    cfg = f"""# OpenEvolve config -- crossword generator evolution
max_iterations: {iterations}
checkpoint_interval: 10
llm:
  models:
    - name: "{model}"
      weight: 1.0
  api_base: "{api_base}"
  api_key: "EMPTY"
  temperature: 0.7
  max_tokens: 8000
database:
  population_size: 300
  num_islands: 3
  archive_size: 60
  migration_interval: 20
  feature_dimensions: ["fill_density", "coverage"]
evaluator:
  timeout: 45
  # serial for now: our harvest appends multi-KB lines to one JSONL, and
  # concurrent appends can interleave. Per-worker harvest files would allow >1.
  parallel_evaluations: 1
  enable_artifacts: true
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(cfg)
    return out_path


def write_problem(spec, edu, path, n_draws=1, seed=0):
    """Per-spec problem context the evaluator reads via OE_PROBLEM_JSON."""
    problem = {
        "spec_id": spec.spec_id,
        "size": spec.size,
        "require_symmetry": spec.require_symmetry,
        "min_word_len": spec.min_word_len,
        "time_budget_s": spec.time_budget_s,
        "density_target": spec.density_target,
        "topic_words": edu["targets"],       # coverage rewards placing the vocab
        "word_source": edu["allowed"],
        "scores": edu["scores"],
        "n_draws": n_draws,
        "seed": seed,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(problem, fh)
    return path


def _run_openevolve_cli(initial_program, config_path, iterations):
    """Invoke OpenEvolve. CONFIRM the exact entry point on Colab against the
    installed version -- this uses the documented CLI form."""
    cmd = [sys.executable, "-m", "openevolve.cli", initial_program, _EVALUATOR,
           "--config", config_path, "--iterations", str(iterations)]
    subprocess.run(cmd, check=True)


def run_spec(spec, edu, run_dir, config_path, initial_program, iterations, dry_run):
    problem_path = os.path.join(run_dir, f"problem_{spec.spec_id}.json")
    harvest_path = os.path.join(run_dir, "harvest.jsonl")
    write_problem(spec, edu, problem_path)
    os.environ[oe_evaluator.PROBLEM_ENV] = problem_path
    os.environ[oe_evaluator.HARVEST_ENV] = harvest_path

    if dry_run:
        # Mimic ONE OpenEvolve candidate: score the seed and harvest it.
        oe_evaluator.evaluate(initial_program)
    else:
        _run_openevolve_cli(initial_program, config_path, iterations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "runs", "pilot"))
    ap.add_argument("--n-specs", type=int, default=15)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="", help="comma-separated pilot grid sizes, e.g. 7,9 (blank = all)")
    ap.add_argument("--seeds", default="reference_v1,csp_ac3,beam_search",
                    help="comma-separated seed generator names in seeds/ (distinct algorithm families)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-common-fill", action="store_true")
    args = ap.parse_args()

    run_dir = args.out
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(run_dir, "dataset")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    specs = generate_specs(args.n_specs, seed=args.seed, **({"sizes": sizes} if sizes else {}))
    save_specs(specs, os.path.join(run_dir, "specs.jsonl"))
    edu = build_education_source(include_common_fill=args.include_common_fill)
    print(f"specs={len(specs)} | palette={edu['n_allowed']:,} "
          f"(vocab={edu['n_vocab']:,}, fill={edu['n_fill']:,})")

    config_path = write_config(os.path.join(run_dir, "config.yaml"), args.model, args.api_base, args.iterations)
    harvest_path = os.path.join(run_dir, "harvest.jsonl")
    if os.path.exists(harvest_path):
        os.remove(harvest_path)

    seed_names = [s.strip() for s in args.seeds.split(",") if s.strip()]
    print(f"seeds: {seed_names}")
    for sname in seed_names:
        seed_path = os.path.join(_SEEDS_DIR, sname + ".py")
        if not os.path.exists(seed_path):
            print(f"  !! seed not found: {seed_path} — skipping")
            continue
        ip = make_initial_program(seed_path, os.path.join(run_dir, f"initial_{sname}.py"))
        for i, spec in enumerate(specs):
            print(f"[seed {sname}] [{i+1}/{len(specs)}] spec {spec.spec_id} (size {spec.size}) "
                  f"{'[dry-run]' if args.dry_run else ''}")
            run_spec(spec, edu, run_dir, config_path, ip, args.iterations, args.dry_run)

    # Process the harvest -> dataset.
    rows = [json.loads(x) for x in open(harvest_path, encoding="utf-8")] if os.path.exists(harvest_path) else []
    specs_by_id = {s.spec_id: s for s in specs}
    out = harvest.process_harvest(rows, specs_by_id)
    counts = build_dataset.build(out["solutions"], data_dir)
    print(f"\nharvested {len(rows)} candidates -> {out['n_solutions']} solutions "
          f"({out['kind_counts']}), {out['n_negatives']} negatives, "
          f"{out['n_distinct_programs']} distinct programs")
    print(f"dataset splits: {counts}  ->  {data_dir}")


if __name__ == "__main__":
    main()
