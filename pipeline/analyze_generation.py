"""Learnings analyzer: turn a generation's harvest into a per-program and
per-heuristic SCORECARD that says what makes a crossword filler good vs bad.

Reads `runs/genK/harvest.jsonl` + `runs/genK/specs.jsonl` + the generation's
`manifest.json` (name -> heuristic tags), aggregates the recorded verifier metrics
per program and per heuristic, writes `runs/genK/scorecard.json`, and prints a
ranked report. Pure Python (no sandbox). This is the signal the teacher reads to
author the next generation and to weight OpenEvolve's hints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.harvest import ast_hash, process_harvest
from pipeline.spec_generator import load_specs


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _manifest_hashes(gen_dir, extra_dirs=("teachers", "seeds")):
    """ast_hash -> {name, heuristics} for each program named in the manifest. Resolves
    each name in gen_dir first, then extra_dirs (so seed/teacher families named in the
    manifest get tagged too). EVOLVE-BLOCK markers are comments, so a wrapped candidate
    hashes the same as its source file."""
    mpath = os.path.join(gen_dir, "manifest.json")
    manifest = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {"programs": []}
    search = [gen_dir] + [d if os.path.isabs(d) else os.path.join(_ROOT, d) for d in extra_dirs]
    out = {}
    for p in manifest.get("programs", []):
        for base in search:
            fp = os.path.join(base, p["name"] + ".py")
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as fh:
                    out[ast_hash(fh.read())] = {"name": p["name"], "heuristics": p.get("heuristics", [])}
                break
    return out, manifest


def per_program(harvest_rows, specs, names):
    """Aggregate raw candidates by program_hash into per-program quality stats."""
    by = {}
    for row in harvest_rows:
        code = row.get("code", "")
        if not code:
            continue
        ph = ast_hash(code)
        m = row.get("metrics", {}) or {}
        spec = specs.get(row.get("spec_id"))
        budget = spec.time_budget_s if spec else 5.0
        d = by.setdefault(ph, {"runs": 0, "valid": [], "filler": [], "invalid": [],
                               "rt_ratio": [], "coverage": [], "combined": []})
        d["runs"] += 1
        d["valid"].append(float(m.get("valid", 0) or 0))
        d["filler"].append(float(m.get("filler_fraction", 0) or 0))
        d["invalid"].append(float(m.get("invalid_crossing_frac", 0) or 0)
                            + float(m.get("invalid_entry_frac", 0) or 0))
        d["rt_ratio"].append(_clip(float(m.get("runtime_s", 0) or 0) / max(0.1, budget)))
        d["coverage"].append(float(m.get("coverage", 0) or 0))
        d["combined"].append(float(m.get("combined_score", 0) or 0))

    stats = []
    for ph, d in by.items():
        vr, fl = _mean(d["valid"]), _mean(d["filler"])
        iv, rr = _mean(d["invalid"]), _mean(d["rt_ratio"])
        cov, comb = _mean(d["coverage"]), _mean(d["combined"])
        # "what makes a good filler": validity dominates; penalize filler + invalid
        # connections; reward coverage; lightly penalize slow fills.
        composite = round(vr - 0.5 * fl - 0.5 * iv + 0.3 * cov - 0.1 * rr, 4)
        info = names.get(ph, {"name": "unknown", "heuristics": []})
        stats.append({"program_hash": ph, "name": info["name"], "heuristics": info["heuristics"],
                      "runs": d["runs"], "validity_rate": vr, "mean_filler": fl,
                      "mean_invalid": iv, "mean_runtime_ratio": rr, "mean_coverage": cov,
                      "mean_combined": comb, "composite": composite})
    stats.sort(key=lambda s: -s["composite"])
    return stats


def per_heuristic(prog_stats):
    agg = {}
    for s in prog_stats:
        for h in s["heuristics"]:
            agg.setdefault(h, []).append(s["composite"])
    rows = [{"heuristic": h, "n_programs": len(v), "mean_composite": round(sum(v) / len(v), 4)}
            for h, v in agg.items()]
    rows.sort(key=lambda r: -r["mean_composite"])
    return rows


def scorecard(run_dir, gen_dir, out_path=None):
    harvest_rows = _load_jsonl(os.path.join(run_dir, "harvest.jsonl"))
    specs = load_specs(os.path.join(run_dir, "specs.jsonl"))
    names, manifest = _manifest_hashes(gen_dir)
    classified = process_harvest(harvest_rows, specs)

    # per-program classification counts (solution / hindsight_* / negative)
    cls = {}
    for s in classified["solutions"]:
        cls.setdefault(s["program_hash"], {})
        cls[s["program_hash"]][s["kind"]] = cls[s["program_hash"]].get(s["kind"], 0) + 1
    for n in classified["negatives"]:
        cls.setdefault(n["program_hash"], {})
        cls[n["program_hash"]]["negative"] = cls[n["program_hash"]].get("negative", 0) + 1

    progs = per_program(harvest_rows, specs, names)
    for p in progs:
        p["kinds"] = cls.get(p["program_hash"], {})
    heur = per_heuristic(progs)

    card = {
        "gen": manifest.get("gen"),
        "n_candidates": len(harvest_rows),
        "n_distinct_programs": classified["n_distinct_programs"],
        "kind_counts": classified["kind_counts"],
        "failure_counts": classified["failure_counts"],
        "per_heuristic": heur,
        "per_program": progs,
    }
    out_path = out_path or os.path.join(run_dir, "scorecard.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    _print(card)
    return card


def _print(card):
    print(f"\n=== scorecard gen {card['gen']}: {card['n_candidates']} candidates, "
          f"{card['n_distinct_programs']} distinct programs ===")
    print(f"kinds: {card['kind_counts']} | failures: {card['failure_counts']}\n")
    if card["per_heuristic"]:
        print("heuristics ranked by mean composite (what makes a good filler):")
        for h in card["per_heuristic"]:
            print(f"  {h['mean_composite']:+.3f}  {h['heuristic']:24s} ({h['n_programs']} programs)")
    print("\ntop programs:")
    for p in card["per_program"][:5]:
        print(f"  {p['composite']:+.3f}  {p['name']:22s} valid={p['validity_rate']:.2f} "
              f"filler={p['mean_filler']:.2f} inv={p['mean_invalid']:.2f} "
              f"cov={p['mean_coverage']:.2f}  {p['kinds']}")
    if len(card["per_program"]) > 5:
        print("bottom programs:")
        for p in card["per_program"][-3:]:
            print(f"  {p['composite']:+.3f}  {p['name']:22s} valid={p['validity_rate']:.2f} "
                  f"filler={p['mean_filler']:.2f} inv={p['mean_invalid']:.2f}  {p['kinds']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with harvest.jsonl + specs.jsonl")
    ap.add_argument("--gen", required=True, help="generation dir with manifest.json + programs")
    args = ap.parse_args()
    scorecard(args.run, args.gen)


if __name__ == "__main__":
    main()
