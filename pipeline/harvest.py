"""Turn the OpenEvolve harvest (every evaluated candidate) into training pairs.

Reads harvest rows {spec_id, code, metrics, artifacts{best_draw}} and the spec
catalog, then produces (spec -> program) SFT rows:

  - SOLUTION            meets the QUALITY bar: is_valid AND filler_fraction <= 0.30
                        AND no invalid connections AND runtime within the budget
  - HINDSIGHT (density) valid but missed a soft bar -> relabel the (soft) density
                        target down to what the program actually achieved
  - HINDSIGHT (symmetry) the ONLY failing check was symmetry -> it's a valid
                        ASYMMETRIC crossword; relabel to a non-symmetric spec
  - NEGATIVE            anything else -> KEPT + labeled (failure_category) and
                        persisted for analysis / optional DPO, not SFT

Programs are de-duplicated by AST hash and capped per program (positives and
negatives separately) so the model can't just memorize a handful of generators.
This is the SOAR "hindsight" idea applied offline over the OpenEvolve trace (no
iterative retrain loop).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace

if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pipeline.spec_generator import render_spec

SYMMETRY_REASON = "black squares not 180-degree symmetric"


def ast_hash(code: str) -> str:
    """Structure-only hash so cosmetic differences (comments/spacing) collapse."""
    try:
        norm = ast.dump(ast.parse(code), annotate_fields=False)
    except SyntaxError:
        norm = " ".join(code.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _best_draw(row):
    return (row.get("artifacts", {}) or {}).get("best_draw", {}) or {}


# A failed sandbox run reports its status as a bare token (reasons == [status]);
# match those EXACTLY (substring matching would e.g. see "oom" inside "boom").
_STATUS_MAP = {
    "timeout": "timeout", "oom": "oom", "exception": "exception",
    "no_function": "malformed", "syntax_error": "malformed",
    "banned_import": "malformed", "bad_schema": "malformed",
}
# Scorer reasons are sentences; match those by substring. Ordered specific-first.
_FAILURE_RULES = [
    ("timeout", ("timed out", "deadline")),
    ("oom", ("out of memory", "memoryerror", "rlimit")),
    ("malformed", ("no generate_crossword", "syntaxerror", "banned import",
                   "bad schema", "missing across", "not iterable")),
    ("empty_grid", ("empty grid",)),
    ("nonword", ("not real words", "nonword", "not a word")),
    ("crossing_conflict", ("conflict", "overlap")),
    ("disconnected", ("not connected", "disconnected", "connected region")),
    ("declared_mismatch", ("declared", "!= actual", "actual horizontal", "actual vertical")),
    ("short_run", ("at least", "shorter than", "min length")),
    ("asymmetry", ("symmetr",)),
    ("out_of_bounds", ("out of bounds", "oob")),
    ("exception", ("traceback", "exception")),
]


def classify_failure(reasons, metrics=None, best_draw=None) -> str:
    """Bucket a negative candidate by its dominant failure mode."""
    reasons = reasons or []
    for r in reasons:                        # exact sandbox-status match first
        rs = str(r).strip().lower()
        if rs in _STATUS_MAP:
            return _STATUS_MAP[rs]
    text = " | ".join(str(r) for r in reasons).lower()
    if not text:
        if metrics and int(round(metrics.get("valid", 0) or 0)) == 1:
            return "low_coverage"   # valid crossword that only missed the quality bar
        return "other"
    for label, needles in _FAILURE_RULES:
        if any(n in text for n in needles):
            return label
    return "other"


def _neg_metrics(m):
    keys = ("valid", "fill_density", "coverage", "filler_fraction",
            "invalid_crossing_frac", "invalid_entry_frac", "runtime_s", "combined_score")
    return {k: m.get(k) for k in keys}


def process_harvest(harvest_rows, specs, hindsight_floor=0.4, per_program_cap=20,
                    per_negative_cap=10, max_filler=0.30) -> dict:
    """harvest_rows: list of harvest dicts. specs: {spec_id: SpecRecord}.

    A candidate is a top-tier `solution` when its crossword meets the user's quality
    bar: is_valid AND filler_fraction <= max_filler AND no invalid connections
    (crossing or entry) AND runtime within the spec's budget. Valid crosswords that
    miss a soft bar are salvaged as hindsight positives (density relabeled down, or
    symmetry relaxed). Everything else is a labeled NEGATIVE -- persisted (not
    discarded) for analysis / optional DPO. combined_score is only a secondary sort
    key now, not the acceptance gate.
    """
    pos_counts: dict = {}
    neg_counts: dict = {}
    solutions, negatives = [], []

    for row in harvest_rows:
        spec = specs.get(row.get("spec_id"))
        code = row.get("code", "")
        if spec is None or not code:
            continue
        m = row.get("metrics", {}) or {}
        bd = _best_draw(row)
        reasons = bd.get("reasons", []) or []
        ph = ast_hash(code)

        # ALL draws valid (mean == 1.0), not just a rounded majority -- with n_draws>1
        # this is the reliability filter: a program must fill on EVERY draw to qualify.
        valid = float(m.get("valid", 0) or 0) >= 0.999
        combined = float(m.get("combined_score", 0.0) or 0.0)
        filler = float(m.get("filler_fraction", 0.0) or 0.0)
        invalid = (float(m.get("invalid_crossing_frac", 0.0) or 0.0)
                   + float(m.get("invalid_entry_frac", 0.0) or 0.0))
        runtime = float(m.get("runtime_s", 0.0) or 0.0)
        good = (valid and filler <= max_filler and invalid == 0.0
                and runtime <= spec.time_budget_s)

        rec, kind = None, None
        if good:
            rec, kind = spec, "solution"
        elif valid and bd.get("fill_density", 0.0) >= hindsight_floor:
            rec = replace(spec, density_target=round(bd.get("fill_density", spec.density_target), 2))
            kind = "hindsight_density"
        elif (not valid) and spec.require_symmetry and reasons == [SYMMETRY_REASON]:
            rec, kind = replace(spec, require_symmetry=False), "hindsight_symmetry"
        else:
            if neg_counts.get(ph, 0) >= per_negative_cap:
                continue
            neg_counts[ph] = neg_counts.get(ph, 0) + 1
            negatives.append({
                "spec_id": spec.spec_id,
                "spec": render_spec(spec),
                "effective_spec": spec.as_dict(),
                "code": code,
                "kind": "negative",
                "split": spec.split,
                "program_hash": ph,
                "combined_score": combined,
                "metrics": _neg_metrics(m),
                "reasons": reasons,
                "failure_category": classify_failure(reasons, m, bd),
            })
            continue

        if pos_counts.get(ph, 0) >= per_program_cap:
            continue
        pos_counts[ph] = pos_counts.get(ph, 0) + 1
        solutions.append({
            "spec_id": spec.spec_id,
            "spec": render_spec(rec),
            "effective_spec": rec.as_dict(),
            "code": code,
            "kind": kind,
            "split": spec.split,
            "combined_score": combined,
            "program_hash": ph,
        })

    return {
        "solutions": solutions,
        "negatives": negatives,
        "n_solutions": len(solutions),
        "n_negatives": len(negatives),
        "n_distinct_programs": len(pos_counts),
        "n_distinct_negatives": len(neg_counts),
        "kind_counts": _count_kinds(solutions),
        "failure_counts": _count_failures(negatives),
    }


def _count_kinds(rows):
    out: dict = {}
    for r in rows:
        out[r["kind"]] = out.get(r["kind"], 0) + 1
    return out


def _count_failures(negatives):
    out: dict = {}
    for r in negatives:
        out[r["failure_category"]] = out.get(r["failure_category"], 0) + 1
    return out
