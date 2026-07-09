"""Consolidate the per-section datasets into single train/dev/eval files for training.

The SFT corpus is split across run dirs (bulk + gen3 = 7/9 construct; templates11 +
templates15 = fixed-template). This unions them, dedups by (spec_id, program_hash),
and writes data/sft/{train,dev,eval}.jsonl. `eval` stays PRISTINE (never trained).

Because 11x11/15x15 are only ~6% of train each, --upsample duplicates those sizes'
TRAIN rows (dev/eval untouched) so the model sees the template behavior often enough
to learn it. Duplicates are exact copies, so keep the factor modest (memorization
risk) -- this is a training-time balance knob, not new data.

    python pipeline/merge_dataset.py --upsample 11=3,15=3
"""

from __future__ import annotations

import argparse
import glob
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SECTIONS = ["runs/bulk", "runs/gen3", "runs/templates15", "runs/templates11"]


def _load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _key(row):
    m = row.get("meta", row)
    return (m.get("spec_id"), m.get("program_hash"))


def _size(row):
    return (row.get("meta", {}).get("effective_spec") or {}).get("size")


def merge_split(sections, split):
    seen, out = set(), []
    for sec in sections:
        for row in _load(os.path.join(sec, "dataset", f"{split}.jsonl")):
            k = _key(row)
            if k in seen:
                continue
            seen.add(k)
            out.append(row)
    return out


def parse_upsample(spec):
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        size, factor = part.split("=")
        out[int(size)] = int(factor)
    return out


def dist(rows):
    from collections import Counter
    return dict(sorted(Counter(_size(r) for r in rows).items()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=DEFAULT_SECTIONS)
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "sft"))
    ap.add_argument("--upsample", default="11=3,15=3",
                    help="comma list size=factor; duplicates those sizes' TRAIN rows")
    args = ap.parse_args(argv)
    sections = [s if os.path.isabs(s) else os.path.join(_ROOT, s) for s in args.sections]
    ups = parse_upsample(args.upsample)

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for split in ("train", "dev", "eval"):
        rows = merge_split(sections, split)
        if split == "train" and ups:
            extra = []
            for r in rows:
                f = ups.get(_size(r), 1)
                extra.extend([r] * (f - 1))       # (f-1) additional copies
            print(f"  train upsample {ups}: +{len(extra)} duplicate rows")
            rows = rows + extra
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[split] = len(rows)
        print(f"  {split}: {len(rows)} rows  sizes={dist(rows)}  -> {path}")

    # negatives (optional DPO); dedup, exclude eval-derived
    for name in ("negatives", "negatives_eval"):
        seen, out = set(), []
        for sec in sections:
            for row in _load(os.path.join(sec, "dataset", f"{name}.jsonl")):
                k = _key(row)
                if k in seen:
                    continue
                seen.add(k); out.append(row)
        if out:
            p = os.path.join(args.out, f"{name}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                for r in out:
                    fh.write(json.dumps(r) + "\n")
            print(f"  {name}: {len(out)} rows -> {p}")

    print(f"\nmerged SFT: train {counts['train']} / dev {counts['dev']} / eval {counts['eval']} "
          f"(eval is held out, never trained)")


if __name__ == "__main__":
    main()
