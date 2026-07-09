"""Fleet eval: query Claude Opus N times (independent, clean-room) and average.

Each of the N samples is an INDEPENDENT Opus generation given ONLY the task rules +
output schema -- NO algorithm hints of any kind (no CSP/MRV/AC-3/beam/templates). This
measures unaugmented Opus at writing a crossword generator, one shot, no iteration.

Every returned program is scored in the SUBPROCESS sandbox (hard timeout/kill, since
the code is untrusted and may hang) against both the English and Spanish palettes, with
every entry validated against a REAL dictionary. Results are averaged across all N.

    python pipeline/eval_opus_fleet.py --n 100 --per-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.sandbox import run_candidate
from harness.scorer import Spec, score
from pipeline.eval_harness import extract_code
from pipeline.eval_selfmodel import BUDGET, TOPICS, english_palette, spanish_palette
from pipeline.eval_selfmodel import _norm as norm

# CLEAN-ROOM contract: rules + schema only. NO technique hints.
CONTRACT = """Write one self-contained Python function that generates a fixed-grid, American-style crossword. Output ONLY the code in a single response.

def generate_crossword(topic: str, word_source, size: int) -> dict

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; black squares in 180-degree rotational symmetry; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15.

Output only the Python code."""

BASE = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
TOK = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
MODEL = os.environ.get("FLEET_MODEL", "claude-opus-4-8")
CTX = ssl._create_unverified_context()


def query_opus(idx, max_tokens=8000, temperature=1.0, retries=4):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "system": "You are an expert Python programmer. When asked for code, output only code.",
            "messages": [{"role": "user", "content": CONTRACT}]}
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


GEN_TIMEOUT = None   # if set, overrides the per-size budget for the runner hard-kill


def score_one(code, pal, size, topic):
    budget = BUDGET.get(size, size * 2)
    spec_d = {"topic": topic, "word_source": pal["ws"], "size": size, "seed": 0}
    res = run_candidate(code, spec_d, timeout_s=(GEN_TIMEOUT or budget), mem_mb=1024)
    z = {"valid": 0, "fully": 0, "within": 0, "dict_frac": 0.0, "coverage": 0.0,
         "crossings": 0, "entries": 0, "filler": 0.0}
    if res["status"] != "ok" or not res.get("result"):
        return z
    lay = res["result"]
    spec = Spec(size=size, topic_words=tuple(pal["targets"]), require_symmetry=True,
                min_word_len=3, time_budget_s=budget)
    try:
        m = score(lay, spec, pal["allowed"], runtime_s=res["runtime_s"], vocab_set=pal["clean_set"])
    except Exception:
        return z
    ents = [e["answer"] for e in (lay.get("across") or []) + (lay.get("down") or [])
            if len(str(e.get("answer", ""))) >= 3]
    df = (sum(1 for w in ents if norm(w) in pal["DICT"]) / len(ents)) if ents else 0.0
    valid = int(m["valid"] == 1)
    filler = m["filler_fraction"] or 0.0
    within = int(valid and filler <= 0.30 and res["runtime_s"] <= budget)
    return {"valid": valid, "fully": int(valid and df >= 0.999), "within": within,
            "dict_frac": df, "coverage": m["coverage"], "crossings": m["crossings"],
            "entries": m["n_entries"], "filler": filler}


def agg(rs):
    n = len(rs) or 1
    v = [r for r in rs if r["valid"]]; vn = len(v) or 1
    return {"n": len(rs), "valid": sum(r["valid"] for r in rs) / n,
            "fully": sum(r["fully"] for r in rs) / n, "within": sum(r["within"] for r in rs) / n,
            "dict": sum(r["dict_frac"] for r in rs) / n, "cov": sum(r["coverage"] for r in v) / vn,
            "cross": sum(r["crossings"] for r in v) / vn, "ent": sum(r["entries"] for r in v) / vn,
            "filler": sum(r["filler"] for r in v) / vn}


def table(title, rows, sizes):
    print(f"\n===== {title} =====")
    hdr = (f"{'size':>5}{'n':>5}{'valid%':>8}{'fullyOK%':>10}{'within%':>9}"
           f"{'dictOK':>8}{'cov':>6}{'cross':>7}{'entries':>8}{'filler%':>9}")
    print(hdr); print("-" * len(hdr))
    for size in sizes:
        a = agg([r for r in rows if r["size"] == size])
        print(f"{size:>5}{a['n']:>5}{a['valid']*100:>7.0f}{a['fully']*100:>9.0f}{a['within']*100:>8.0f}"
              f"{a['dict']*100:>7.0f}{a['cov']:>6.2f}{a['cross']:>7.0f}{a['ent']:>8.0f}{a['filler']*100:>8.0f}")
    a = agg(rows); print("-" * len(hdr))
    print(f"{'ALL':>5}{a['n']:>5}{a['valid']*100:>7.0f}{a['fully']*100:>9.0f}{a['within']*100:>8.0f}"
          f"{a['dict']*100:>7.0f}{a['cov']:>6.2f}{a['cross']:>7.0f}{a['ent']:>8.0f}{a['filler']*100:>8.0f}")
    return agg(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of independent Opus samples")
    ap.add_argument("--per-size", type=int, default=1, help="topics scored per size per program")
    ap.add_argument("--api-workers", type=int, default=8)
    ap.add_argument("--score-workers", type=int, default=8)
    ap.add_argument("--gen-timeout", type=float, default=None,
                    help="override per-size time budget for the runner hard-kill (seconds); "
                         "use with the no-'few seconds' prompt to give programs generous time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not BASE or not TOK:
        sys.exit("ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN not set")
    if a.gen_timeout:
        global GEN_TIMEOUT
        GEN_TIMEOUT = a.gen_timeout
        print(f"generous runner timeout: {GEN_TIMEOUT}s (per-size budget overridden)")

    print(f"querying {MODEL} x {a.n} (clean-room, no technique hints) @ {BASE}", flush=True)
    progs, done = [], 0
    with ThreadPoolExecutor(max_workers=a.api_workers) as ex:
        futs = [ex.submit(query_opus, i) for i in range(a.n)]
        for f in as_completed(futs):
            progs.append(f.result()); done += 1
            if done % 10 == 0:
                ok = sum(1 for c in progs if c)
                print(f"  generated {done}/{a.n} ({ok} parsed as code)", flush=True)
    parsed = [c for c in progs if c]
    print(f"parse rate: {len(parsed)}/{a.n} = {100*len(parsed)/a.n:.0f}%", flush=True)
    if not parsed:
        sys.exit("no programs parsed")

    print("building EN + ES palettes...", flush=True)
    pals = {"en": (english_palette(15), [7, 9, 11, 15]), "es": (spanish_palette(11), [7, 9, 11])}
    results = {}
    for lang, (pal, sizes) in pals.items():
        print(f"scoring {len(parsed)} programs on {lang.upper()} (sizes {sizes})...", flush=True)
        tasks = [(c, size, t) for c in parsed for size in sizes for t in TOPICS[lang][:a.per_size]]
        rows = []
        with ThreadPoolExecutor(max_workers=a.score_workers) as ex:
            futs = {ex.submit(score_one, c, pal, size, t): size for (c, size, t) in tasks}
            for f in as_completed(futs):
                rec = f.result(); rec["size"] = futs[f]; rows.append(rec)
        results[lang] = {"rows": rows, "sizes": sizes}

    summary = {"model": MODEL, "n_samples": a.n, "parse_rate": len(parsed) / a.n, "by_lang": {}}
    for lang in ("en", "es"):
        ov = table(f"Claude Opus (unaugmented, n={a.n} samples) — {lang.upper()}",
                   results[lang]["rows"], results[lang]["sizes"])
        summary["by_lang"][lang] = {"overall": ov,
                                    "by_size": {s: agg([r for r in results[lang]["rows"] if r["size"] == s])
                                                for s in results[lang]["sizes"]}}
    print("\nfullyOK% = structurally valid AND every entry a real dictionary word")

    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "runs", "eval", f"opus_fleet_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
