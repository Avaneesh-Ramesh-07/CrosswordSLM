"""Regenerate the non-hardcoded SFT dataset onto the NEW prompt, folding in gen4, with a
full re-verification that re-sorts every record positive/negative.

What it does:
  1. Assemble every candidate program from the current `data/sft/` (positives
     train/dev/eval + negatives) AND the gen4 section (`runs/gen4/dataset/`), recovering
     the raw program (stripping any legacy contract-comment header). Dedup (spec_id, hash).
  2. Re-verify EVERYTHING on `WORD_LIST_FULLY_PURIFIED` as word_source, `--runs` times per
     distinct (program_hash, size) with a different topic each run (different randomized
     grid). Reclassify by the strict validity gate: VALID on ALL runs (structure + real
     words + no invalid crossings, filler <= 0.30) -> positive `solution`; else -> labeled
     negative (`failure_category` via harvest.classify_failure). Runtime is measured and
     reported (within budget) but NOT gated, to avoid parallelism-driven runtime noise.
  3. Serialize with the new prompt (build_dataset.to_chat -> assistant = program only,
     user = the shared contract; negatives keep the flat schema with spec = new prompt)
     into `data/sft_non_hardcoded_enhanced/`. `data/sft/` is left untouched (the backup).

    python pipeline/regen_sft.py [--out data/sft_non_hardcoded_enhanced] [--runs 3] [--workers 12]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate
from harness.scorer import Spec, score
from pipeline.build_dataset import build, write_negatives, CONTRACT_COMMENT
from pipeline.eval_harness import extract_code
from pipeline.eval_selfmodel import BUDGET
from pipeline.harvest import classify_failure

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WL = os.path.join(_ROOT, "data", "wordlists")
TOPICS = ["vocabulary", "words", "study", "learn", "review"]  # one per run, forces distinct grids


def load_purified():
    """The ONE and ONLY word list is WORD_LIST_FULLY_PURIFIED (topic is always 'vocabulary').
    Deliberately does NOT load words_alpha or sat_words: validity is judged SOLELY against this
    list (a word not in it is invalid), and the word_source handed to programs is split into
    theme (>=4-letter vocabulary) + fill (3-letter glue) entirely from within this same list."""
    words = sorted({l.strip().upper() for l in open(os.path.join(_WL, "WORD_LIST_FULLY_PURIFIED.txt"), encoding="utf-8")
                    if l.strip() and l.strip().isalpha()})
    allowed = set(words)
    theme = [w for w in words if len(w) >= 4]
    fill = [w for w in words if len(w) == 3]
    return {"ws": {"theme": theme, "fill": fill}, "allowed": allowed, "clean_set": allowed,
            "targets": theme, "DICT": allowed}


def _strip_contract(code: str) -> str:
    code = (code or "").strip()
    if code.startswith(CONTRACT_COMMENT):
        code = code[len(CONTRACT_COMMENT):].lstrip("\n")
    return code


def load_positives(path, default_split):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        m = r.get("meta", {}) or {}
        es = m.get("effective_spec") or {}
        code = _strip_contract(extract_code(r["messages"][2]["content"]) or "")
        out.append({"spec_id": m.get("spec_id"), "program_hash": m.get("program_hash"),
                    "size": es.get("size"), "split": m.get("split") or es.get("split") or default_split,
                    "code": code, "effective_spec": es, "orig": "positive"})
    return out


def load_negatives(path, default_split):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        es = r.get("effective_spec") or {}
        out.append({"spec_id": r.get("spec_id"), "program_hash": r.get("program_hash"),
                    "size": es.get("size") or r.get("size"), "split": r.get("split") or default_split,
                    "code": r.get("code", ""), "effective_spec": es, "orig": "negative"})
    return out


def assemble():
    recs = []
    for sec in ("data/sft", "runs/gen4/dataset"):
        recs += load_positives(os.path.join(_ROOT, sec, "train.jsonl"), "train")
        recs += load_positives(os.path.join(_ROOT, sec, "dev.jsonl"), "dev")
        recs += load_positives(os.path.join(_ROOT, sec, "eval.jsonl"), "eval")
        recs += load_negatives(os.path.join(_ROOT, sec, "negatives.jsonl"), "train")
        recs += load_negatives(os.path.join(_ROOT, sec, "negatives_eval.jsonl"), "eval")
    # dedup by (spec_id, program_hash) — keep first (code identical for same hash)
    seen, deduped = set(), []
    for r in recs:
        if not r.get("program_hash") or not r.get("code") or not r.get("size"):
            continue
        k = (r["spec_id"], r["program_hash"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    return deduped


def verify_one(code, size, pal, runs):
    """Run the program `runs` times on the purified palette; return verdict dict."""
    budget = BUDGET.get(size, size * 2)
    timeout_s = max(30.0, budget + 15.0)
    spec = Spec(size=size, topic_words=tuple(pal["targets"]), require_symmetry=False,
                min_word_len=3, time_budget_s=budget)
    per = []
    for i in range(runs):
        # topic is ALWAYS "vocabulary" (per the contract); cross-run variation comes from the
        # per-process hash randomization of hash((topic,size)) + unseeded RNG in the generators.
        res = run_candidate(code, {"topic": "vocabulary", "word_source": pal["ws"], "size": size, "seed": i},
                            timeout_s=timeout_s, mem_mb=1024)
        if res.get("status") != "ok" or not res.get("result"):
            per.append({"good": False, "status": res.get("status", "error"),
                        "reasons": [res.get("status", "error")], "m": {}, "rt": res.get("runtime_s", 0.0)})
            continue
        try:
            m = score(res["result"], spec, pal["allowed"], runtime_s=res["runtime_s"], vocab_set=pal["clean_set"])
        except Exception as e:
            per.append({"good": False, "status": "score_error", "reasons": [f"score error: {e}"],
                        "m": {}, "rt": res.get("runtime_s", 0.0)})
            continue
        invalid = (m.get("invalid_crossing_frac", 0.0) or 0.0) + (m.get("invalid_entry_frac", 0.0) or 0.0)
        good = (m.get("valid") == 1 and invalid == 0.0 and (m.get("filler_fraction") or 0.0) <= 0.30)
        per.append({"good": bool(good), "status": "ok", "reasons": m.get("reasons", []) or [],
                    "m": m, "rt": res["runtime_s"], "within": res["runtime_s"] <= budget})
    valid_count = sum(1 for p in per if p["good"])
    good_ref = next((p for p in per if p["good"]), None)
    fail_ref = next((p for p in per if not p["good"]), None)
    return {"valid_count": valid_count, "runs": len(per), "per": per,
            "good_ref": good_ref, "fail_ref": fail_ref}


def load_extra_programs(dirs, sizes, reps):
    """Fold in raw *.py generators (e.g. gen5 fusions) as fresh candidate records, routed to
    `sizes` with `reps` training records each. They are re-verified like everything else."""
    from pipeline.harvest import ast_hash
    out = []
    for d in [x.strip() for x in dirs.split(",") if x.strip()]:
        for pth in sorted(glob.glob(os.path.join(_ROOT, d, "*.py"))):
            base = os.path.splitext(os.path.basename(pth))[0]
            if base.startswith("_"):
                continue
            code = open(pth, encoding="utf-8").read()
            if "def generate_crossword" not in code:
                continue
            ph = ast_hash(code)
            for sz in sizes:
                for r in range(reps):
                    sid = f"{base}_s{sz}_{r:03d}"
                    out.append({"spec_id": sid, "program_hash": ph, "size": sz, "split": "train",
                                "code": code, "orig": "new",
                                "effective_spec": {"spec_id": sid, "size": sz, "require_symmetry": False,
                                                   "min_word_len": 3, "time_budget_s": BUDGET.get(sz, sz * 2),
                                                   "split": "train"}})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_non_hardcoded_enhanced")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="keep <= physical cores; oversubscription starves programs' internal deadlines")
    ap.add_argument("--min-valid", type=int, default=0,
                    help="POSITIVE if valid in >= this many of --runs (0 = require ALL runs)")
    ap.add_argument("--extra-prog-dirs", default="",
                    help="comma list of dirs of raw *.py generators to fold in (e.g. generations/gen5)")
    ap.add_argument("--extra-reps", type=int, default=24, help="training records per (extra program, size)")
    ap.add_argument("--extra-sizes", default="7,9,11", help="sizes to route extra programs to")
    a = ap.parse_args(argv)
    out_dir = os.path.join(_ROOT, a.out)
    mv = a.min_valid if a.min_valid > 0 else a.runs

    pal = load_purified()
    print(f"purified palette: allowed={len(pal['allowed'])} theme={len(pal['ws']['theme'])} "
          f"fill={len(pal['ws']['fill'])} DICT={len(pal['DICT'])}", flush=True)

    recs = assemble()
    if a.extra_prog_dirs:
        extra = load_extra_programs(a.extra_prog_dirs, [int(s) for s in a.extra_sizes.split(",")], a.extra_reps)
        recs += extra
        print(f"folded in {len(extra)} extra records from {a.extra_prog_dirs}", flush=True)
    by_orig = collections.Counter(r["orig"] for r in recs)
    print(f"assembled {len(recs)} records (orig positive={by_orig['positive']}, negative={by_orig['negative']}, "
          f"new={by_orig['new']}); gate: valid in >= {mv}/{a.runs} runs", flush=True)

    # distinct (program_hash, size) -> code ; verify each `runs` times
    pairs = {}
    for r in recs:
        pairs.setdefault((r["program_hash"], r["size"]), r["code"])
    print(f"verifying {len(pairs)} distinct (program, size) pairs x {a.runs} runs on the purified palette...", flush=True)

    verdicts, done = {}, 0
    def work(item):
        (ph, sz), code = item
        return (ph, sz), verify_one(code, sz, pal, a.runs)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for key, v in ex.map(work, list(pairs.items())):
            verdicts[key] = v
            done += 1
            if done % 25 == 0:
                nv = sum(1 for x in verdicts.values() if x["valid_count"] >= mv)
                print(f"  {done}/{len(pairs)} verified ({nv} pass >= {mv}/{a.runs})", flush=True)

    # assign each record + build solution / negative rows
    solutions, negatives = [], []
    moved = collections.Counter()
    for r in recs:
        v = verdicts[(r["program_hash"], r["size"])]
        positive = v["valid_count"] >= mv
        ref = (v["good_ref"] if positive else v["fail_ref"]) or v["per"][0]
        m = ref.get("m", {}) or {}
        es = {**(r["effective_spec"] or {}), "topic": "vocabulary"}  # every entry is topic 'vocabulary'
        if positive:
            solutions.append({
                "spec_id": r["spec_id"], "spec": "", "effective_spec": es,
                "code": r["code"], "kind": "solution",
                "combined_score": m.get("combined_score", 0.0), "program_hash": r["program_hash"],
                "split": r["split"],
            })
            moved["neg->pos" if r["orig"] == "negative" else ("new->pos" if r["orig"] == "new" else "pos->pos")] += 1
        else:
            negatives.append({
                "spec_id": r["spec_id"], "spec": "", "effective_spec": es,
                "code": r["code"], "kind": "negative", "split": r["split"],
                "program_hash": r["program_hash"], "combined_score": m.get("combined_score", 0.0),
                "metrics": {k: m.get(k) for k in ("valid", "fill_density", "coverage", "filler_fraction",
                            "invalid_crossing_frac", "invalid_entry_frac")} | {"runtime_s": ref.get("rt")},
                "reasons": ref.get("reasons", []),
                "failure_category": classify_failure(ref.get("reasons", []), m, None),
            })
            moved["pos->neg" if r["orig"] == "positive" else ("new->neg" if r["orig"] == "new" else "neg->neg")] += 1

    counts = build(solutions, out_dir)
    negc = write_negatives(negatives, out_dir)

    # ---- report ----
    def is_pos(r):
        return verdicts[(r["program_hash"], r["size"])]["valid_count"] >= mv
    print(f"\n=== reclassified on the purified palette (POSITIVE = valid in >= {mv}/{a.runs} runs) ===")
    print(f"  orig positive -> negative (demoted): {moved['pos->neg']}")
    print(f"  orig negative -> positive (promoted): {moved['neg->pos']}")
    print(f"  new (extra) programs -> positive: {moved['new->pos']}  | -> negative: {moved['new->neg']}")
    print(f"  final positives: {counts} (total {sum(counts.values())})")
    print(f"  final negatives: {negc}")
    print("  positives by size:", dict(sorted(collections.Counter(r["size"] for r in recs if is_pos(r)).items())))
    print("  negatives by size:", dict(sorted(collections.Counter(r["size"] for r in recs if not is_pos(r)).items())))
    print(f"\n  wrote enhanced dataset to {a.out}/ ; data/sft/ left untouched (backup)")


if __name__ == "__main__":
    main()
