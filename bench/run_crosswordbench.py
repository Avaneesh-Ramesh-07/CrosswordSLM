"""Run the CrossWordBench eval against a model (or a baseline) and aggregate.

The "model" is pluggable so the same harness scores real and baseline systems:

  --model reference        score each puzzle's OWN reference grid (sanity/ceiling
                           for crossings; shows references are NOT scorer-valid)
  --model seed:NAME        use a seed generator (seeds/NAME.py) as the model
  --model endpoint         query an OpenAI-compatible server for a program, given
                           the SPEC prompt (reuses the query_qwen setup)

For program models the generated code is run in the harness sandbox with
word_source = the puzzle's exact word set, size = the puzzle size; the returned
layout is scored by bench.crosswordbench.score_layout. Metrics are aggregated
overall and per grid size.

Examples:
    python3 bench/run_crosswordbench.py --model reference --split 7x7
    python3 bench/run_crosswordbench.py --model seed:reference_v1 --limit 10
    python3 bench/run_crosswordbench.py --model endpoint \
        --base-url http://localhost:8000/v1 --model-name Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import urllib.error
import urllib.request

if __package__ in (None, ""):
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.crosswordbench import load_puzzles, score_layout
from harness.scorer import build_layout_from_grid
from harness.sandbox import run_candidate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "data", "crosswordbench")
_SEEDS = os.path.join(_ROOT, "seeds")

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


# --- model adapters: each maps a Puzzle -> a layout dict (or None) ------------

def _reference_layout(puzzle, raw_grid):
    grid = {(r, c): ch for r, row in enumerate(raw_grid)
            for c, ch in enumerate(row) if ch != "-"}
    return build_layout_from_grid(grid, puzzle.size), None


def _run_program(code, puzzle, timeout_s):
    run = run_candidate(
        code,
        {"topic": "general vocabulary", "word_source": list(puzzle.words),
         "size": puzzle.size, "seed": 0},
        timeout_s=timeout_s,
    )
    return run.get("result"), run.get("runtime_s")


def _seed_code(name: str) -> str:
    path = os.path.join(_SEEDS, f"{name}.py")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_code(text: str) -> str:
    # Qwen3 is a reasoning model: drop any <think> block, then take the last
    # fenced code block (models often narrate before the final answer).
    text = _THINK.sub("", text or "").strip()
    blocks = _CODE_FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def _extract_json(text: str) -> dict:
    text = _THINK.sub("", text or "").strip()
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        text = blocks[-1].strip()
    m = _JSON_OBJ.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return {}


def _chat(system: str, user: str, base_url: str, model_name: str, api_key: str,
          max_tokens: int = 3072) -> str:
    """One OpenAI-compatible chat completion; returns the message content."""
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # vLLM honors this to disable Qwen3 thinking; dropped on 400 below.
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def _post(body: dict) -> dict:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key or 'x'}"})
        with urllib.request.urlopen(req, timeout=240) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        data = _post(payload)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        payload.pop("chat_template_kwargs", None)  # server rejects the extra field
        data = _post(payload)
    return data["choices"][0]["message"]["content"]


# "/no_think" is a harmless no-op on non-Qwen servers.
_SYS_PROGRAM = ("You output only one self-contained Python program defining "
                "generate_crossword(topic, word_source, size) that returns the layout "
                "dict. No prose. /no_think")
_SYS_DIRECT = ("You output only a single JSON object describing the crossword layout. "
               "No prose, no code. /no_think")


def _endpoint_code(prompt, base_url, model_name, api_key):
    return _extract_code(_chat(_SYS_PROGRAM, prompt, base_url, model_name, api_key))


def _endpoint_layout(prompt, base_url, model_name, api_key):
    return _extract_json(_chat(_SYS_DIRECT, prompt, base_url, model_name, api_key))


# --- aggregation --------------------------------------------------------------

def _agg(results: list) -> dict:
    if not results:
        return {}
    def mean(key, cond=lambda r: True):
        xs = [r[key] for r in results if cond(r) and r[key] is not None]
        return round(st.mean(xs), 4) if xs else 0.0
    n = len(results)
    return {
        "n": n,
        "success_rate": round(sum(r["success"] for r in results) / n, 4),
        "mean_crossings": mean("crossings"),
        "mean_crossings_vs_ref": mean("crossings_vs_ref"),
        "mean_crossing_ratio": mean("crossing_ratio"),
        "mean_coverage": mean("coverage"),
        "used_all_words_rate": round(sum(r["used_all_words"] for r in results) / n, 4),
        "mean_abs_black_delta": round(st.mean(abs(r["black_delta"]) for r in results), 4),
        "status_counts": _counts(r["status"] for r in results),
    }


def _counts(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _print_report(model, overall, by_size):
    print(f"\n===== CrossWordBench eval :: model={model} =====")
    print(f"puzzles: {overall.get('n', 0)}")
    print(f"SUCCESS RATE (valid config)     : {overall.get('success_rate')}")
    print(f"mean crossings                  : {overall.get('mean_crossings')}")
    print(f"mean crossings / reference      : {overall.get('mean_crossings_vs_ref')}")
    print(f"mean crossing ratio (checked)   : {overall.get('mean_crossing_ratio')}")
    print(f"mean word coverage              : {overall.get('mean_coverage')}")
    print(f"used-all-words rate             : {overall.get('used_all_words_rate')}")
    print(f"mean |black-square delta|       : {overall.get('mean_abs_black_delta')}")
    print(f"status counts                   : {overall.get('status_counts')}")
    for size, agg in sorted(by_size.items()):
        print(f"  [size {size}] n={agg['n']} success={agg['success_rate']} "
              f"crossings={agg['mean_crossings']} vs_ref={agg['mean_crossings_vs_ref']} "
              f"coverage={agg['mean_coverage']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reference | seed:NAME | endpoint")
    ap.add_argument("--mode", default="program", choices=("program", "direct"),
                    help="endpoint output: program (run in sandbox) or direct layout JSON")
    ap.add_argument("--relaxed", action="store_true",
                    help="CrossWordBench-style validity (unchecked cells allowed) vs NYT-strict")
    ap.add_argument("--config", default="english")
    ap.add_argument("--split", default=None, help="e.g. 7x7 or 14x14 (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="max puzzles per split")
    ap.add_argument("--out", default=None, help="write per-puzzle results JSONL here")
    ap.add_argument("--base-url", default=os.environ.get("OE_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model-name", default=os.environ.get("OE_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    args = ap.parse_args()

    pattern = f"{args.config}_{args.split}.jsonl" if args.split else f"{args.config}_*.jsonl"
    paths = sorted(glob.glob(os.path.join(_DATA, pattern)))
    if not paths:
        raise SystemExit(f"no data files match {pattern} in {_DATA}")

    seed_code = _seed_code(args.model.split(":", 1)[1]) if args.model.startswith("seed:") else None
    results = []
    for path in paths:
        # raw rows kept in parallel so --model reference can rebuild the grid
        raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        puzzles = load_puzzles(path)
        if args.limit:
            raw, puzzles = raw[:args.limit], puzzles[:args.limit]
        for row, pz in zip(raw, puzzles):
            runtime_s = None
            if args.model == "reference":
                grid = json.loads(row["puzzle_state"])["grid"]
                layout, runtime_s = _reference_layout(pz, grid)
            elif seed_code is not None:
                layout, runtime_s = _run_program(seed_code, pz, timeout_s=float(max(8, pz.size)))
            elif args.model == "endpoint":
                if args.mode == "direct":
                    layout = _endpoint_layout(pz.direct_prompt(), args.base_url,
                                              args.model_name, args.api_key)
                else:
                    code = _endpoint_code(pz.prompt(), args.base_url, args.model_name, args.api_key)
                    layout, runtime_s = _run_program(code, pz, timeout_s=float(max(8, pz.size)))
            else:
                raise SystemExit(f"unknown --model {args.model}")
            results.append(score_layout(layout, pz, runtime_s=runtime_s, relaxed=args.relaxed))

    overall = _agg(results)
    by_size = {size: _agg([r for r in results if r["size"] == size])
               for size in sorted({r["size"] for r in results})}
    label = f"{args.model}:{args.mode}" if args.model == "endpoint" else args.model
    label += " [relaxed]" if args.relaxed else " [strict]"
    _print_report(label, overall, by_size)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} per-puzzle rows -> {args.out}")


if __name__ == "__main__":
    main()
