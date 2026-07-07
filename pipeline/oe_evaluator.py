"""OpenEvolve evaluator + SOAR harvest hook.

OpenEvolve evolves crossword-generator PROGRAMS and calls `evaluate(program_path)`
each iteration. We wrap our safe sandbox + deterministic scorer as that evaluator:
run the candidate on several fresh word_source draws (fuzz-verify, so lucky-seed
overfits don't win), score it, and return metrics with `combined_score` as the
fitness OpenEvolve optimizes. `fill_density` and `coverage` are exposed as
MAP-Elites feature dimensions to force quality-diversity.

Every evaluated candidate is appended to a harvest JSONL (the SOAR trace we later
turn into (spec -> program) training pairs), so we capture 100% of candidates
regardless of OpenEvolve's own population eviction.

The core `evaluate_code(...)` is pure and unit-testable without OpenEvolve. The
`evaluate(program_path)` entry point (what OpenEvolve calls) reads the run's
problem context and harvest path from env vars and wraps the result.
"""

from __future__ import annotations

import json
import os
import random
import time

# Allow running/importing directly.
if __package__ in (None, ""):
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec
from harness.verify import fuzz_verify

# OpenEvolve is only installed in the run environment (Colab); optional here.
try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:  # pragma: no cover - openevolve not installed locally
    EvaluationResult = None

PROBLEM_ENV = "OE_PROBLEM_JSON"     # path to this run's problem context
HARVEST_ENV = "OE_HARVEST_JSONL"    # path to append harvested candidates to


def make_draws(spec: Spec, word_source, n_draws=1, seed=0, cap=None):
    """Build `n_draws` (spec, word_source) draws.

    With cap=None each draw uses the FULL word source (reliable fill — this is
    what OpenEvolve fitness uses). Pass a `cap` to subsample for a robustness
    check; each draw always includes the spec's topic_words so coverage stays
    achievable. NOTE: subsampling can under-fill maximally-constrained grids
    (e.g. an open word square), so the strong anti-hardcode gate lives at
    dataset-harvest time on genuinely different word sources, not here.
    """
    if isinstance(word_source, dict):
        # theme+fill contract: pass the structured source straight to the generator
        # (fuzz_verify flattens theme+fill for the scorer).
        return [(spec, word_source) for _ in range(n_draws)]
    ws = list(word_source)
    targets = list(spec.topic_words)
    use_full = cap is None or cap >= len(ws)
    draws = []
    for i in range(n_draws):
        if use_full:
            subset = sorted(set(ws) | set(targets))
        else:
            rng = random.Random(seed * 1000 + i)
            subset = sorted(set(rng.sample(ws, cap)) | set(targets))
        draws.append((spec, subset))
    return draws


def _mean(results, key):
    if not results:
        return 0.0
    return round(sum(r.get(key, 0.0) or 0.0 for r in results) / len(results), 4)


def evaluate_code(code: str, spec: Spec, word_source, scores=None, n_draws=1, seed=0, cap=None) -> dict:
    """Run + score a candidate across draws. Returns {metrics, artifacts, fuzz}."""
    draws = make_draws(spec, word_source, n_draws=n_draws, seed=seed, cap=cap)
    timeout_s = max(spec.time_budget_s * 2.0, spec.time_budget_s + 3.0)
    fuzz = fuzz_verify(code, draws, scores=scores, timeout_s=timeout_s, mem_mb=1536)
    results = fuzz["results"]

    metrics = {
        "combined_score": fuzz["mean_score"],   # OpenEvolve fitness
        "valid": _mean(results, "valid"),        # validity rate across draws
        "fill_density": _mean(results, "fill_density"),   # feature dim
        "coverage": _mean(results, "coverage"),           # feature dim
        "fill_quality": _mean(results, "fill_quality"),
        "runtime_ok": _mean(results, "runtime_ok"),
        "worst_score": fuzz["min_score"],
    }
    failed = [r for r in results if r.get("valid") != 1]
    best = max(results, key=lambda r: r.get("combined_score", 0.0)) if results else {}
    artifacts = {
        "n_valid": fuzz["n_valid"],
        "n_draws": fuzz["n"],
        "failed_checks": " | ".join(str(r.get("reasons")) for r in failed)[:2000],
        # best_draw diagnostics drive SOAR hindsight relabeling at harvest time
        "best_draw": {
            "valid": best.get("valid", 0),
            "symmetry_ok": best.get("symmetry_ok"),
            "connected": best.get("connected"),
            "fill_density": best.get("fill_density", 0.0),
            "coverage": best.get("coverage", 0.0),
            "reasons": best.get("reasons", ["did not run"]),
        },
    }
    return {"metrics": metrics, "artifacts": artifacts, "fuzz": fuzz}


# --- OpenEvolve entry point --------------------------------------------------

def _load_problem(path):
    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)
    # theme+fill contract: topic_words = the theme vocabulary (what coverage rewards)
    theme = p.get("theme")
    topic_words = theme if theme is not None else p.get("topic_words", ())
    spec = Spec(
        size=p["size"],
        topic_words=tuple(topic_words),
        require_symmetry=p.get("require_symmetry", True),
        min_word_len=p.get("min_word_len", 3),
        time_budget_s=p.get("time_budget_s", 5.0),
        density_target=p.get("density_target", 0.72),
    )
    return p, spec


def _harvest(path, spec_id, code, out):
    if not path:
        return
    row = {
        "spec_id": spec_id,
        "code": code,
        "metrics": out["metrics"],
        "artifacts": out["artifacts"],
        "ts": time.time(),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def evaluate(program_path):
    """Entry point OpenEvolve calls. Reads the run's problem context + harvest
    path from env, scores the candidate, logs it, and returns metrics."""
    problem, spec = _load_problem(os.environ[PROBLEM_ENV])
    with open(program_path, encoding="utf-8") as fh:
        code = fh.read()

    theme = problem.get("theme")
    if theme is not None:
        word_source = {"theme": theme, "fill": problem.get("fill", [])}
    else:
        word_source = problem["word_source"]
    out = evaluate_code(
        code, spec,
        word_source=word_source,
        scores=problem.get("scores"),
        n_draws=problem.get("n_draws", 1),
        seed=problem.get("seed", 0),
        cap=problem.get("cap"),
    )
    _harvest(os.environ.get(HARVEST_ENV), problem.get("spec_id", "?"), code, out)

    metrics, artifacts = out["metrics"], out["artifacts"]
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics
