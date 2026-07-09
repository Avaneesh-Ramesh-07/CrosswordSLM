"""Persistent elite library (MAP-Elites), cumulative across generations.

Keeps the best valid program per (coverage, filler_fraction) feature cell in
`data/elites/` (`index.json` + `programs/<hash>.py`). Two consumers:
  (a) the next LOCAL generation branches from `export_seeds(...)`;
  (b) the real OpenEvolve run on Colab uses the same exports as seeds / warm-starts.

This is the piece that lets "what worked" carry forward into future generations
(the AlphaEvolve/OpenEvolve elite-archive idea), used only to build the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.harvest import ast_hash

FEATURE = ("coverage", "filler_fraction")   # matches the OpenEvolve MAP-Elites axes


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x or 0.0)))


def feature_cell(metrics):
    """Bucket a candidate into a MAP-Elites cell (0.1 resolution on each axis)."""
    return f"cov={round(_clip(metrics.get('coverage')), 1):.1f}|fil={round(_clip(metrics.get('filler_fraction')), 1):.1f}"


def _paths(elites_dir):
    return os.path.join(elites_dir, "index.json"), os.path.join(elites_dir, "programs")


def load_elites(elites_dir):
    idx_path, _ = _paths(elites_dir)
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def update_elites(harvest_rows, elites_dir, gen, min_valid=1.0):
    """Insert valid candidates; keep the highest combined_score per feature cell."""
    idx_path, prog_dir = _paths(elites_dir)
    os.makedirs(prog_dir, exist_ok=True)
    index = load_elites(elites_dir)
    added = 0
    for row in harvest_rows:
        code = row.get("code", "")
        m = row.get("metrics", {}) or {}
        if not code or float(m.get("valid", 0) or 0) < min_valid:
            continue
        cell = feature_cell(m)
        combined = float(m.get("combined_score", 0) or 0)
        cur = index.get(cell)
        if cur is None or combined > cur["combined_score"]:
            ph = ast_hash(code)
            fname = f"{ph}.py"
            with open(os.path.join(prog_dir, fname), "w", encoding="utf-8") as fh:
                fh.write(code)
            index[cell] = {
                "program_hash": ph, "gen": gen, "combined_score": round(combined, 4),
                "coverage": round(float(m.get("coverage", 0) or 0), 4),
                "filler_fraction": round(float(m.get("filler_fraction", 0) or 0), 4),
                "fill_density": round(float(m.get("fill_density", 0) or 0), 4),
                "runtime_s": round(float(m.get("runtime_s", 0) or 0), 4),
                "code_file": fname,
            }
            added += 1
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return {"n_cells": len(index), "n_added": added}


def export_seeds(elites_dir, out_seeds_dir, top_k=8):
    """Copy the top-k elites (by combined_score) into out_seeds_dir as elite_NN.py.
    Returns the seed-name list for --seeds / --seeds-dir consumers."""
    _, prog_dir = _paths(elites_dir)
    index = load_elites(elites_dir)
    ranked = sorted(index.values(), key=lambda e: -e["combined_score"])[:top_k]
    os.makedirs(out_seeds_dir, exist_ok=True)
    names = []
    for i, e in enumerate(ranked):
        src = os.path.join(prog_dir, e["code_file"])
        if not os.path.exists(src):
            continue
        name = f"elite_{i:02d}"
        with open(src, encoding="utf-8") as fh:
            code = fh.read()
        with open(os.path.join(out_seeds_dir, name + ".py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elites", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "elites"))
    args = ap.parse_args()
    index = load_elites(args.elites)
    print(f"{len(index)} elite cells in {args.elites}")
    for cell, e in sorted(index.items(), key=lambda kv: -kv[1]["combined_score"]):
        print(f"  {cell:22s} combined={e['combined_score']:.3f} gen={e['gen']} "
              f"cov={e['coverage']:.2f} fil={e['filler_fraction']:.2f} -> {e['code_file']}")


if __name__ == "__main__":
    main()
