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
from pipeline.spec_generator import generate_specs, load_hint_weights, save_specs
from pipeline.word_source import build_clean_education_source, build_education_source

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


# Default quality rule injected into OpenEvolve's system prompt. The generation
# orchestrator overrides `system_message` with scorecard-derived winning heuristics.
DEFAULT_SYSTEM_MESSAGE = (
    "You are evolving a Python crossword generator. Maximize valid, fully-interlocked "
    "grids where EVERY entry is a real word taken from word_source (never invented), "
    "and MINIMIZE filler: prefer common, gettable vocabulary over crosswordese. Strong "
    "constraint propagation (AC-3/MAC) is what makes large grids fill; layer other "
    "heuristics on top of it."
)


def write_config(out_path, model, api_base, iterations, feature_dims=None, system_message=None):
    """Write an OpenEvolve config.yaml. NOTE: confirm keys against the installed
    openevolve version on Colab; adjust if v0.3.x renamed anything.

    feature_dims: MAP-Elites axes (default promotes a QUALITY axis, filler_fraction,
    alongside coverage). system_message: injected into the evolution prompt so
    learnings steer generation."""
    dims = feature_dims or ["coverage", "filler_fraction"]
    dims_yaml = "[" + ", ".join(f'"{d}"' for d in dims) + "]"
    msg = (system_message or DEFAULT_SYSTEM_MESSAGE).replace('"', "'")
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
prompt:
  system_message: "{msg}"
database:
  population_size: 300
  num_islands: 3
  archive_size: 60
  migration_interval: 20
  feature_dimensions: {dims_yaml}
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


def write_problem(spec, edu, path, n_draws=1, seed=0, in_process=False):
    """Per-spec problem context the evaluator reads via OE_PROBLEM_JSON."""
    problem = {
        "spec_id": spec.spec_id,
        "size": spec.size,
        "require_symmetry": spec.require_symmetry,
        "min_word_len": spec.min_word_len,
        "time_budget_s": spec.time_budget_s,
        "density_target": spec.density_target,
        # theme+fill contract: generator gets {theme, fill}; scorer rewards theme (coverage)
        "theme": edu["targets"],
        "fill": edu.get("fill_words", []),
        "scores": edu["scores"],
        # surface the spec's heuristic hints so the OpenEvolve LLM (not just the SFT
        # user turn) sees which techniques to try -- the evaluator echoes them back
        "heuristic_hints": list(spec.heuristic_hints),
        "n_draws": n_draws,
        "seed": seed,
        "in_process": in_process,   # fast non-sandboxed eval for trusted teacher code
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


def run_spec(spec, edu, run_dir, config_path, initial_program, iterations, dry_run,
             n_draws=1, in_process=False):
    problem_path = os.path.join(run_dir, f"problem_{spec.spec_id}.json")
    harvest_path = os.path.join(run_dir, "harvest.jsonl")
    write_problem(spec, edu, problem_path, n_draws=n_draws, in_process=in_process)
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
                    help="comma-separated seed generator names (distinct algorithm families)")
    ap.add_argument("--seeds-dir", default="seeds,teachers",
                    help="comma-separated dirs (relative to repo root) to resolve --seeds names from, "
                         "e.g. generations/gen1,data/elites_seeds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-common-fill", action="store_true")
    ap.add_argument("--clean-palette", action="store_true",
                    help="use the clean educational palette (wordfreq n dict n crossword); "
                         "answers are 100%% real gettable vocabulary (filler~0)")
    ap.add_argument("--hint-weights", default="",
                    help="path to a prior generation's scorecard.json to bias hint sampling "
                         "(carries learnings forward)")
    ap.add_argument("--n-draws", type=int, default=1,
                    help="evaluate each candidate on N draws; a solution must be valid on ALL "
                         "of them (reliability filter). N>1 re-runs to catch timing flakiness")
    ap.add_argument("--in-process", action="store_true",
                    help="fast non-sandboxed eval (TRUSTED teacher code only): no subprocess "
                         "spawn, no word_source re-serialization")
    ap.add_argument("--seed-sizes", default="",
                    help="per-seed size routing 'name:7,9;name2:11' -- skip (seed,spec) whose "
                         "size the seed can't fill, avoiding wasted doomed attempts")
    args = ap.parse_args()

    # parse size routing: {seed_name: {allowed sizes}}
    seed_sizes = {}
    for part in args.seed_sizes.split(";"):
        if ":" in part:
            name, szs = part.split(":", 1)
            seed_sizes[name.strip()] = {int(s) for s in szs.split(",") if s.strip()}

    run_dir = args.out
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(run_dir, "dataset")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    hint_weights = load_hint_weights(args.hint_weights) if (args.hint_weights and os.path.exists(args.hint_weights)) else None
    gen_kwargs = {"hint_weights": hint_weights}
    if sizes:
        gen_kwargs["sizes"] = sizes
    specs = generate_specs(args.n_specs, seed=args.seed, **gen_kwargs)
    save_specs(specs, os.path.join(run_dir, "specs.jsonl"))
    if hint_weights:
        print(f"hint weights loaded from {args.hint_weights}")
    if args.clean_palette:
        edu = build_clean_education_source()
        edu.setdefault("n_fill", len(edu["fill_words"]))
    else:
        edu = build_education_source(include_common_fill=args.include_common_fill)
    print(f"specs={len(specs)} | palette={edu['n_allowed']:,} "
          f"(vocab={edu['n_vocab']:,}, fill={edu['n_fill']:,})")

    config_path = write_config(os.path.join(run_dir, "config.yaml"), args.model, args.api_base, args.iterations)
    harvest_path = os.path.join(run_dir, "harvest.jsonl")
    if os.path.exists(harvest_path):
        os.remove(harvest_path)

    seed_dirs = [d.strip() for d in args.seeds_dir.split(",") if d.strip()]

    def resolve_seed(sname):
        for d in seed_dirs:
            base = d if os.path.isabs(d) else os.path.join(_ROOT, d)
            p = os.path.join(base, sname + ".py")
            if os.path.exists(p):
                return p
        return None

    seed_names = [s.strip() for s in args.seeds.split(",") if s.strip()]
    print(f"seeds: {seed_names}  (dirs: {seed_dirs})")
    for sname in seed_names:
        seed_path = resolve_seed(sname)
        if seed_path is None:
            print(f"  !! seed not found: {sname} — skipping")
            continue
        ip = make_initial_program(seed_path, os.path.join(run_dir, f"initial_{sname}.py"))
        allowed = seed_sizes.get(sname)   # None -> all sizes
        for i, spec in enumerate(specs):
            if allowed is not None and spec.size not in allowed:
                continue   # size routing: this seed can't fill this size -> skip the doomed run
            print(f"[seed {sname}] [{i+1}/{len(specs)}] spec {spec.spec_id} (size {spec.size}) "
                  f"{'[dry-run]' if args.dry_run else ''}")
            run_spec(spec, edu, run_dir, config_path, ip, args.iterations, args.dry_run,
                     n_draws=args.n_draws, in_process=args.in_process)

    # Process the harvest -> dataset.
    rows = [json.loads(x) for x in open(harvest_path, encoding="utf-8")] if os.path.exists(harvest_path) else []
    specs_by_id = {s.spec_id: s for s in specs}
    out = harvest.process_harvest(rows, specs_by_id)
    counts = build_dataset.build(out["solutions"], data_dir)
    neg = build_dataset.write_negatives(out["negatives"], data_dir)
    print(f"\nharvested {len(rows)} candidates -> {out['n_solutions']} solutions "
          f"({out['kind_counts']}), {out['n_negatives']} negatives "
          f"({out['failure_counts']}), {out['n_distinct_programs']} distinct programs")
    print(f"dataset splits: {counts}  (DPO negatives={neg['pool']}, held-out eval negatives={neg['eval']})  ->  {data_dir}")


if __name__ == "__main__":
    main()
