"""FAIR "own-terms" eval: run each base-Opus bare-prompt program AS-IS (its own interface)
and judge the crossword it actually produces -- NOT conformance to our generate_crossword
API. Confirms whether 0% is a real failure or an API artifact.

Each of N independent Opus sessions gets the bare eval.jsonl prompt. Its program is saved,
run as a subprocess (own interface), and classified by what it actually outputs:
  crashed | hung | no_code | skipped_unsafe | FAILED_TO_FILL (no-solution/blank/errors) |
  FILLED_CANDIDATE (printed a filled grid -> flagged for manual verification)
All programs + outputs are written to runs/eval/asis/ for inspection.

    python pipeline/eval_opus_asis.py --per-size 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import extract_code

BASE = os.environ["ANTHROPIC_BASE_URL"].rstrip("/"); TOK = os.environ["ANTHROPIC_AUTH_TOKEN"]
CTX = ssl._create_unverified_context(); MODEL = "claude-opus-4-8"
OUTDIR = "runs/eval/asis"

# markers that mean the program's OWN run did NOT produce a filled crossword
FAIL_MARKERS = [
    "no solution", "not found", "no valid", "add more words", "could not", "couldn't",
    "unable", "failed", "skipped", "out of bounds", "conflict", "empty grid",
    "blank grid", "no fill", "incomplete", "give up", "gave up", "retry", "error",
    "complete: false", "complete:false", "complete = false", "complete=false",
    "not complete", "unsolved", "partial", "cannot", "warning",
]
# refuse to execute programs that touch the system/network (untrusted code)
UNSAFE = re.compile(r"\b(subprocess|socket|shutil|requests|urllib|os\.system|os\.remove|"
                    r"os\.rmdir|rmtree|__import__|eval\(|exec\()|open\([^)]*['\"][wa]")


def call(user):
    body = {"model": MODEL, "max_tokens": 14000, "temperature": 1.0,
            "system": "You are an expert Python programmer.",
            "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(BASE + "/v1/messages", data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                 "authorization": f"Bearer {TOK}"}, method="POST")
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=200, context=CTX) as r:
                d = json.loads(r.read())
            return "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
        except Exception:
            continue
    return ""


def load_prompts(path, sizes, per_size, seed=0):
    by = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l); s = d["meta"]["effective_spec"]["size"]
        by.setdefault(s, []).append(d["messages"][1]["content"])
    rng = random.Random(seed)
    return [(rng.choice(by[s]), s) for s in sizes if by.get(s) for _ in range(per_size)]


def run_asis(i, code, size):
    if not code:
        return {"i": i, "size": size, "verdict": "no_code", "out": ""}
    path = os.path.join(OUTDIR, f"prog_{i:03d}_s{size}.py")
    open(path, "w", encoding="utf-8").write(code)
    if UNSAFE.search(code):
        return {"i": i, "size": size, "verdict": "skipped_unsafe", "out": ""}
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=35, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"i": i, "size": size, "verdict": "hung", "out": ""}
    out, err = (p.stdout or ""), (p.stderr or "")
    open(os.path.join(OUTDIR, f"out_{i:03d}_s{size}.txt"), "w", encoding="utf-8").write(out + "\n---STDERR---\n" + err)
    if p.returncode != 0 or "traceback" in err.lower():
        return {"i": i, "size": size, "verdict": "crashed", "out": out}
    low = out.lower()
    if any(m in low for m in FAIL_MARKERS):
        return {"i": i, "size": size, "verdict": "failed_to_fill", "out": out}
    # letters arranged in grid rows (a possible actual fill) -> flag for manual check
    letters = len(re.findall(r"[A-Z]", out))
    return {"i": i, "size": size,
            "verdict": "FILLED_CANDIDATE" if letters >= size * 3 else "no_grid_output",
            "out": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default="data/sft/eval.jsonl")
    ap.add_argument("--sizes", default="7,9,11,15")
    ap.add_argument("--per-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    sizes = [int(s) for s in a.sizes.split(",")]
    os.makedirs(OUTDIR, exist_ok=True)
    prompts = load_prompts(a.eval_file, sizes, a.per_size)
    print(f"as-is own-terms eval: {len(prompts)} bare prompts, {MODEL}", flush=True)

    def work(item):
        user, size = item
        return run_asis(prompts.index(item) if item in prompts else id(item), extract_code(call(user)), size)

    recs, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(lambda it=(u, s), idx=i: run_asis(idx, extract_code(call(it[0])), it[1])): i
                for i, (u, s) in enumerate(prompts)}
        for f in as_completed(futs):
            recs.append(f.result()); done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(prompts)} run", flush=True)

    from collections import Counter
    print("\n=== outcome distribution (running each program on its OWN terms) ===")
    overall = Counter(r["verdict"] for r in recs)
    for size in sizes:
        c = Counter(r["verdict"] for r in recs if r["size"] == size)
        print(f"  size {size}: {dict(c)}")
    print(f"  ALL (n={len(recs)}): {dict(overall)}")
    cand = [r for r in recs if r["verdict"] == "FILLED_CANDIDATE"]
    print(f"\nFILLED_CANDIDATE (need manual validity check): {len(cand)} -> indices {[r['i'] for r in cand]}")
    print(f"all programs + outputs saved under {OUTDIR}/")
    json.dump({"model": MODEL, "n": len(recs), "overall": dict(overall),
               "by_size": {s: dict(Counter(r["verdict"] for r in recs if r["size"] == s)) for s in sizes},
               "records": [{k: r[k] for k in ("i", "size", "verdict")} for r in recs]},
              open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
