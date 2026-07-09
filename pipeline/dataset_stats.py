"""Report dataset entry counts per generation + cumulative (deduped).

Each generation writes its own runs/genK/dataset/. The FINAL fine-tuning dataset is
the union across generations, deduped so re-running a program on the same spec
doesn't double-count. Dedup key = (spec_id, program_hash).

    python pipeline/dataset_stats.py --runs runs/gen1 runs/gen2
"""

from __future__ import annotations

import argparse
import json
import os


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sft_key(row):
    m = row.get("meta", {})
    return (m.get("spec_id"), m.get("program_hash"))


def _neg_key(row):
    return (row.get("spec_id"), row.get("program_hash"))


def report(run_dirs):
    cum = {k: set() for k in ("train", "dev", "eval", "neg", "neg_eval")}
    hdr = f"{'generation':20s} {'train':>6s} {'dev':>5s} {'eval':>5s} {'neg':>6s} {'neg_eval':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for rd in run_dirs:
        d = os.path.join(rd, "dataset")
        row = {}
        for split in ("train", "dev", "eval"):
            rows = _load(os.path.join(d, f"{split}.jsonl"))
            row[split] = len(rows)
            cum[split].update(_sft_key(r) for r in rows)
        negs = _load(os.path.join(d, "negatives.jsonl"))
        neval = _load(os.path.join(d, "negatives_eval.jsonl"))
        row["neg"], row["neg_eval"] = len(negs), len(neval)
        cum["neg"].update(_neg_key(r) for r in negs)
        cum["neg_eval"].update(_neg_key(r) for r in neval)
        name = os.path.basename(os.path.normpath(rd))
        print(f"{name:20s} {row['train']:6d} {row['dev']:5d} {row['eval']:5d} {row['neg']:6d} {row['neg_eval']:9d}")

    print("-" * len(hdr))
    print(f"{'CUMULATIVE (dedup)':20s} {len(cum['train']):6d} {len(cum['dev']):5d} "
          f"{len(cum['eval']):5d} {len(cum['neg']):6d} {len(cum['neg_eval']):9d}")
    sft = len(cum["train"]) + len(cum["dev"]) + len(cum["eval"])
    print(f"\nSFT examples (train+dev+eval, deduped): {sft}")
    print(f"DPO negatives pool: {len(cum['neg'])}   |   held-out eval negatives: {len(cum['neg_eval'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs, e.g. runs/gen1 runs/gen2")
    report(ap.parse_args().runs)


if __name__ == "__main__":
    main()
