"""Base-Opus eval on the held-out eval.jsonl specs, using the STORED (bare) prompts.

Unlike eval_opus_fleet (which sends a fixed clean-room contract), this feeds each Opus
call the EXACT system+user stored in eval.jsonl -- i.e. the minimal system prompt and
the bare user request, with NO contract. That is the true apples-to-apples baseline for
the eventual tuned model (which is evaluated on the same bare prompt): the contract lives
in the tuned model's weights, not the prompt. Base Opus, lacking it, is expected to fail.

Programs are scored in the subprocess sandbox against the clean English palette, with a
real-dictionary check on every entry. Averaged across N generations, balanced by size.

    python pipeline/eval_opus_evalset.py --per-size 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import extract_code
from pipeline.eval_opus_fleet import BASE, MODEL, TOK, agg, score_one, table
from pipeline.eval_selfmodel import english_palette

CTX = ssl._create_unverified_context()


def call(system, user, max_tokens=12000, temperature=1.0, retries=4):
    # Standard mode: fast and reliably emits code. The bare eval.jsonl prompt has NO
    # "answer fast / one-shot" instruction, so Opus is not artificially rushed -- but we
    # do NOT force xhigh adaptive thinking: at 100x it was far too slow and routinely
    # spent the entire token budget thinking, emitting no code at all.
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}]}
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01",
         "authorization": f"Bearer {TOK}"}
    data = json.dumps(body).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/v1/messages", data=data, headers=h, method="POST")
            with urllib.request.urlopen(req, timeout=240, context=CTX) as r:
                d = json.loads(r.read())
            txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
            return extract_code(txt)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < retries - 1:
                time.sleep(2 ** a + 1); continue
            return None
        except Exception:
            if a < retries - 1:
                time.sleep(2 ** a + 1); continue
            return None
    return None


def load_prompts(path, sizes, per_size, seed=0):
    by_size = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        s = d["meta"]["effective_spec"]["size"]
        by_size.setdefault(s, []).append((d["messages"][0]["content"], d["messages"][1]["content"], s))
    rng = random.Random(seed)
    out = []
    for s in sizes:
        pool = by_size.get(s, [])
        if not pool:
            continue
        for _ in range(per_size):
            out.append(rng.choice(pool))       # bare (system, user) drawn from eval.jsonl
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default="data/sft/eval.jsonl")
    ap.add_argument("--sizes", default="7,9,11,15")
    ap.add_argument("--per-size", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not BASE or not TOK:
        sys.exit("ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN not set")
    sizes = [int(s) for s in a.sizes.split(",")]

    prompts = load_prompts(a.eval_file, sizes, a.per_size)
    print(f"eval-set base-Opus run: {len(prompts)} bare prompts from {a.eval_file} "
          f"({MODEL}, temp 1.0)", flush=True)
    print(f"  example bare prompt -> system={prompts[0][0]!r}  user={prompts[0][1]!r}", flush=True)
    pal = english_palette(max(sizes))

    def work(item):
        system, user, size = item
        code = call(system, user)
        if not code:
            z = {"valid": 0, "fully": 0, "within": 0, "dict_frac": 0.0, "coverage": 0.0,
                 "crossings": 0, "entries": 0, "filler": 0.0}
            z["size"] = size
            z["parsed"] = 0
            return z
        rec = score_one(code, pal, size, "vocabulary")
        rec["size"] = size
        rec["parsed"] = 1
        return rec

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, p) for p in prompts]
        for f in as_completed(futs):
            rows.append(f.result()); done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(prompts)} scored", flush=True)

    parse_rate = sum(r.get("parsed", 0) for r in rows) / len(rows)
    print(f"\nparse rate (emitted runnable-looking code): {parse_rate*100:.0f}%")
    ov = table(f"Base {MODEL} on eval.jsonl BARE prompts (n={len(rows)})", rows, sizes)

    out = a.out or f"runs/eval/opus_evalset_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": MODEL, "condition": "bare eval.jsonl prompts", "n": len(rows),
               "parse_rate": parse_rate, "overall": ov,
               "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in sizes}},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
