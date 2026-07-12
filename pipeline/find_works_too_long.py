"""Split the demoted (empty-grid) programs into 'works_too_long' vs genuinely stuck.

A demoted program returned an EMPTY grid because it hit its OWN internal wall-clock deadline
(derived from the per-size budget) before completing a fill on the 24.5k-word purified palette.
To ask "would it finish if we relaxed the time limit?", we run it in a child process with a
patched `time` module so it *perceives* time passing at `dilation`x real speed -> its internal
N-second deadline now takes N/dilation real seconds, i.e. it gets many times its normal search
window. If it then returns a VALID crossword (scored against WORD_LIST_FULLY_PURIFIED only), it
is 'works_too_long' (correct but slow); if it still empties / errors / hangs, it is genuinely stuck.

Programs that qualify are moved OUT of negatives into a new subsection
`data/sft_non_hardcoded_enhanced/works_too_long.jsonl` (chat format, meta.kind='works_too_long',
with the real seconds it needed). Sizes 7/9/11 (15x15 excluded: its 30 s budget would need
minutes of real time per run).

    python pipeline/find_works_too_long.py [--dilation 0.15] [--timeout 100] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score
from pipeline.build_dataset import to_chat
from pipeline.eval_selfmodel import BUDGET
from pipeline.regen_sft import load_purified

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = "data/sft_non_hardcoded_enhanced"


def _wtl_worker(code, size, ws, dilation, q):
    """Child: patch time to run at `dilation`x real speed, then fill once."""
    import time as _t
    rp, rt = _t.perf_counter, _t.time
    p0, t0 = rp(), rt()
    _t.perf_counter = lambda: p0 + (rp() - p0) * dilation
    _t.time = lambda: t0 + (rt() - t0) * dilation
    try:
        _t.monotonic = lambda: p0 + (rp() - p0) * dilation
    except Exception:
        pass
    try:
        ns = {"__name__": "__wtl__"}
        exec(compile(code, "<prog>", "exec"), ns)
        fn = ns.get("generate_crossword")
        if fn is None:
            q.put({"error": "no generate_crossword"}); return
        start = rp()
        lay = fn("vocabulary", ws, size)
        q.put({"layout": lay, "real_s": round(rp() - start, 1)})
    except Exception as e:
        q.put({"error": f"{type(e).__name__}: {str(e)[:80]}"})


def run_dilated(code, size, ws, dilation, timeout):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_wtl_worker, args=(code, size, ws, dilation, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"error": "hung (used full dilated budget)"}
    try:
        return q.get_nowait()
    except Exception:
        return {"error": "no output"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dilation", type=float, default=0.15, help="perceived/real time ratio (0.15 ~= up to 6-20x more search)")
    ap.add_argument("--timeout", type=float, default=100.0, help="real seconds hard cap per run")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only test the first N distinct pairs")
    ap.add_argument("--sizes", default="7,9,11", help="which sizes to scan (already-moved sizes are skipped since they left negatives)")
    a = ap.parse_args(argv)
    SIZES = {int(s) for s in a.sizes.split(",")}

    pal = load_purified()
    ws = pal["ws"]
    # distinct demoted (program_hash, size) with code, sizes 7/9/11
    negs = [json.loads(l) for f in ("negatives", "negatives_eval")
            for l in open(os.path.join(_ROOT, D, f + ".jsonl"), encoding="utf-8")]
    pairs = {}
    for n in negs:
        sz = (n.get("effective_spec") or {}).get("size")
        if sz in SIZES and n.get("program_hash") and n.get("code"):
            pairs.setdefault((n["program_hash"], sz), n["code"])
    items = list(pairs.items())
    if a.limit:
        items = items[:a.limit]
    print(f"testing {len(items)} distinct demoted (program,size) pairs at dilation={a.dilation} "
          f"(~{1/a.dilation:.0f}x more search), real cap {a.timeout:g}s\n", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    wtl = {}          # (hash,size) -> perceived_s  (works given more time)
    still = {}        # (hash,size) -> reason  (genuinely stuck)

    def work(item):
        (ph, sz), code = item
        out = run_dilated(code, sz, ws, a.dilation, a.timeout)
        if "layout" in out and isinstance(out["layout"], dict):
            spec = Spec(size=sz, topic_words=tuple(pal["targets"]), require_symmetry=False,
                        min_word_len=3, time_budget_s=BUDGET.get(sz, sz * 2))
            try:
                m = score(out["layout"], spec, pal["allowed"], runtime_s=0.0, vocab_set=pal["clean_set"])
            except Exception as e:
                return (ph, sz), None, f"score error: {e}"
            inv = (m.get("invalid_crossing_frac", 0) or 0) + (m.get("invalid_entry_frac", 0) or 0)
            if m.get("valid") == 1 and inv == 0:
                return (ph, sz), round(out["real_s"] * a.dilation, 2), None   # perceived fill-time
            return (ph, sz), None, (m.get("reasons") or ["invalid"])[0]
        return (ph, sz), None, out.get("error", "no layout")

    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (ph, sz), perceived, reason in ex.map(work, items):
            done += 1
            if perceived is not None:
                wtl[(ph, sz)] = perceived
                b = BUDGET.get(sz, sz * 2)
                kind = "SLOW" if perceived > b else "flaky"
                print(f"  [{done}/{len(items)}] {ph[:8]} s{sz}: WORKS ({kind}, filled at {perceived}s perceived / budget {b}s)", flush=True)
            else:
                still[(ph, sz)] = reason
                print(f"  [{done}/{len(items)}] {ph[:8]} s{sz}: stuck: {reason}", flush=True)

    print(f"\n=== works_too_long: {len(wtl)} pairs finish given more time; {len(still)} still stuck ===")
    print("  works_too_long pairs:", sorted((h[:8], s, r) for (h, s), r in wtl.items()))

    if a.limit:
        print("\n(smoke run --limit; not writing the subsection)"); return

    # move qualifying records out of negatives into works_too_long.jsonl
    wtl_keys = set(wtl.keys())
    wtl_rows, new_negs = [], []
    for n in negs:
        sz = (n.get("effective_spec") or {}).get("size")
        k = (n.get("program_hash"), sz)
        if k in wtl_keys:
            es = {**(n.get("effective_spec") or {}), "topic": "vocabulary"}
            row = to_chat({"spec_id": n["spec_id"], "kind": "works_too_long", "combined_score": n.get("combined_score", 0.0),
                           "program_hash": n["program_hash"], "effective_spec": es, "split": n.get("split", "train"),
                           "code": n["code"]})
            row["meta"]["works_too_long_real_s"] = wtl[k]
            wtl_rows.append(row)
        else:
            new_negs.append(n)

    # APPEND to any existing works_too_long.jsonl (don't clobber records from a prior size scan)
    wtl_path = os.path.join(_ROOT, D, "works_too_long.jsonl")
    existing = [json.loads(l) for l in open(wtl_path, encoding="utf-8")] if os.path.exists(wtl_path) else []
    with open(wtl_path, "w", encoding="utf-8") as fh:
        for r in existing + wtl_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  (kept {len(existing)} existing + added {len(wtl_rows)} new works_too_long records)")
    # rewrite negatives (minus the moved ones), preserving the eval split
    pool = [n for n in new_negs if n.get("split") != "eval"]
    held = [n for n in new_negs if n.get("split") == "eval"]
    for name, rows in (("negatives.jsonl", pool), ("negatives_eval.jsonl", held)):
        with open(os.path.join(_ROOT, D, name), "w", encoding="utf-8") as fh:
            for n in rows:
                fh.write(json.dumps(n) + "\n")

    bysize = collections.Counter(r["meta"]["effective_spec"]["size"] for r in wtl_rows)
    print(f"\nwrote {len(wtl_rows)} records -> {D}/works_too_long.jsonl (by size: {dict(sorted(bysize.items()))})")
    print(f"negatives now: pool={len(pool)} eval={len(held)} (moved {len(wtl_rows)} out)")


if __name__ == "__main__":
    main()
