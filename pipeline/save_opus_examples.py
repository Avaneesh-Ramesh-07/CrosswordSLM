"""Query Opus a few times (same clean-room prompt) and save one VALID + one INVALID
generated program as inspectable examples, each with the eval verdict in its header.

    python pipeline/save_opus_examples.py --n 30
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate
from harness.scorer import Spec, score
from pipeline.eval_opus_fleet import MODEL, query_opus
from pipeline.eval_selfmodel import BUDGET, english_palette

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_ROOT, "docs", "eval_examples")


def score_prog(code, pal, sizes):
    out = {}
    for size in sizes:
        budget = BUDGET[size]
        spec_d = {"topic": "vocabulary", "word_source": pal["ws"], "size": size, "seed": 0}
        res = run_candidate(code, spec_d, timeout_s=budget)
        if res["status"] != "ok" or not res.get("result"):
            out[size] = {"valid": 0, "reasons": [f"runner: {res['status']}"], "rt": res.get("runtime_s", 0.0)}
            continue
        spec = Spec(size=size, topic_words=tuple(pal["targets"]), require_symmetry=True,
                    min_word_len=3, time_budget_s=budget)
        m = score(res["result"], spec, pal["allowed"], runtime_s=res["runtime_s"], vocab_set=pal["clean_set"])
        out[size] = {"valid": m["valid"], "reasons": list(m["reasons"]), "coverage": m["coverage"],
                     "entries": m["n_entries"], "filler": m["filler_fraction"] or 0.0,
                     "rt": round(res["runtime_s"], 2)}
    return out


def header(kind, res_by_size):
    lines = [f"# ==== {kind} unaugmented {MODEL} generation (clean-room, one-shot) ====",
             "# Verdict from pipeline/eval_selfmodel.py scoring (English clean palette):"]
    for size, r in sorted(res_by_size.items()):
        if r["valid"]:
            lines.append(f"#   {size}x{size}: VALID  entries={r['entries']} cov={r['coverage']:.2f} "
                         f"filler={r['filler']*100:.0f}% rt={r['rt']}s")
        else:
            why = r["reasons"][0] if r["reasons"] else "invalid"
            lines.append(f"#   {size}x{size}: INVALID -> {why}")
    lines.append("# (word_source is injected at call time by the eval; this program uses it as its argument.)")
    return "\n".join(lines) + "\n\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sizes", default="7,9,11")
    a = ap.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",")]
    pal = english_palette(max(sizes))

    print(f"querying {MODEL} x {a.n} for examples...", flush=True)
    progs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(query_opus, i) for i in range(a.n)]):
            c = f.result()
            if c:
                progs.append(c)
    print(f"parsed {len(progs)}/{a.n}; scoring at sizes {sizes}...", flush=True)

    valid_ex = invalid_ex = None
    for code in progs:
        r = score_prog(code, pal, sizes)
        any_valid = any(v["valid"] for v in r.values())
        if any_valid and valid_ex is None:
            valid_ex = (code, r)
        elif not any_valid and invalid_ex is None:
            invalid_ex = (code, r)
        if valid_ex and invalid_ex:
            break

    os.makedirs(OUTDIR, exist_ok=True)
    written = []
    if valid_ex:
        p = os.path.join(OUTDIR, "opus_valid.py")
        open(p, "w", encoding="utf-8").write(header("VALID", valid_ex[1]) + valid_ex[0].strip() + "\n")
        written.append(p); print(f"  wrote {p}")
    else:
        print("  (no valid example found in this batch; increase --n)")
    if invalid_ex:
        p = os.path.join(OUTDIR, "opus_invalid.py")
        open(p, "w", encoding="utf-8").write(header("INVALID", invalid_ex[1]) + invalid_ex[0].strip() + "\n")
        written.append(p); print(f"  wrote {p}")

    # print the verdicts so we can quote them in GAP_ANALYSIS
    for label, ex in (("VALID", valid_ex), ("INVALID", invalid_ex)):
        if ex:
            print(f"\n{label} example verdict:")
            for size, r in sorted(ex[1].items()):
                print(f"  {size}x{size}: valid={r['valid']} reasons={r.get('reasons')}")
    return written


if __name__ == "__main__":
    main()
