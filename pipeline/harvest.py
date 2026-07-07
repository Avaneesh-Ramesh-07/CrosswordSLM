"""Turn the OpenEvolve harvest (every evaluated candidate) into training pairs.

Reads harvest rows {spec_id, code, metrics, artifacts{best_draw}} and the spec
catalog, then produces (spec -> program) SFT rows:

  - SOLUTION            valid AND combined_score >= accept_threshold, for its spec
  - HINDSIGHT (density) valid but lower quality -> relabel the (soft) density
                        target down to what the program actually achieved
  - HINDSIGHT (symmetry) the ONLY failing check was symmetry -> it's a valid
                        ASYMMETRIC crossword; relabel to a non-symmetric spec
  - NEGATIVE            anything else -> reserved for later DPO, not SFT

Programs are de-duplicated by AST hash and capped per program so the model can't
just memorize a handful of generators. This is the SOAR "hindsight" idea applied
offline over the OpenEvolve trace (no iterative retrain loop).
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


def process_harvest(harvest_rows, specs, accept_threshold=0.85, hindsight_floor=0.4,
                    per_program_cap=20) -> dict:
    """harvest_rows: list of harvest dicts. specs: {spec_id: SpecRecord}."""
    counts: dict = {}
    solutions, negatives = [], []

    for row in harvest_rows:
        spec = specs.get(row.get("spec_id"))
        code = row.get("code", "")
        if spec is None or not code:
            continue
        m = row.get("metrics", {})
        valid = int(round(m.get("valid", 0))) == 1
        combined = float(m.get("combined_score", 0.0))
        bd = _best_draw(row)
        reasons = bd.get("reasons", []) or []

        rec, kind = None, None
        if valid and combined >= accept_threshold:
            rec, kind = spec, "solution"
        elif valid and bd.get("fill_density", 0.0) >= hindsight_floor:
            rec = replace(spec, density_target=round(bd.get("fill_density", spec.density_target), 2))
            kind = "hindsight_density"
        elif (not valid) and spec.require_symmetry and reasons == [SYMMETRY_REASON]:
            rec, kind = replace(spec, require_symmetry=False), "hindsight_symmetry"
        else:
            negatives.append({"spec_id": spec.spec_id, "code": code,
                              "reasons": reasons, "combined_score": combined})
            continue

        ph = ast_hash(code)
        if counts.get(ph, 0) >= per_program_cap:
            continue
        counts[ph] = counts.get(ph, 0) + 1
        solutions.append({
            "spec_id": spec.spec_id,
            "spec": render_spec(rec),
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
        "n_distinct_programs": len(counts),
        "kind_counts": _count_kinds(solutions),
    }


def _count_kinds(rows):
    out: dict = {}
    for r in rows:
        out[r["kind"]] = out.get(r["kind"], 0) + 1
    return out
