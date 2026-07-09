"""Base-vs-tuned eval on the held-out (10%) eval split.

Methodology (per project spec):
  * Take the held-out eval specs (never trained/tuned on).
  * Run each spec's (system + user) prompt through a model; the model must emit a
    self-contained generate_crossword(...) program.
  * Run that program through the SAME sandbox + scorer used at harvest time.
  * Report beyond binary success. A crossword is only interesting if it is valid
    AND well-filled, so we grade:
      - within_spec  : valid (size/symmetry/min-run/all-checked/connected/real-words)
                       AND filler <= 30% AND no invalid connections AND in time budget
      - is_valid     : structural validity alone (pass@1)
      - connections  : # across+down entries + # interlocking (checked) cells + coverage
      - non-vocab use : filler_fraction (answers outside the educational palette) and
                        invalid_entry_frac (answers that are not real words at all)

Models are called over an OpenAI-compatible /chat/completions endpoint, so the same
adapter drives Ollama's local qwen3:4b (the base-Qwen baseline) and OpenAI's gpt-5.5
(the frontier baseline). A future fine-tuned checkpoint plugs in as another model.
Model output is UNTRUSTED -> scoring uses the subprocess sandbox (in_process=False).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec
from pipeline.oe_evaluator import evaluate_code
from pipeline.word_source import build_clean_education_source

# --- model registry ----------------------------------------------------------
# Two baselines by default: the frontier model (gpt-5.5) and the base SLM
# (qwen3:4b via local Ollama). token_param/supports_temperature accommodate the
# GPT-5 family (which wants max_completion_tokens and only default temperature).
DEFAULT_MODELS = {
    # base SLM baseline: local Ollama, native API with think=false (thinking mode
    # rambles ~280s/call and never emits code -- disabled for a usable baseline).
    "qwen3-4b-base": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "qwen3:4b",
        "think": False,
        "num_ctx": 8192,
        "concurrency": 2,      # Ollama serializes; small parallelism only
    },
    # frontier baseline: OpenAI. High concurrency -> API calls fire in parallel.
    "gpt-5.5": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.5",
        "api_key_env": "OPEN_AI_API_KEY",
        "supports_temperature": False,
        "token_param": "max_completion_tokens",
        "concurrency": 10,     # OpenAI handles many concurrent requests
    },
}


def load_env(path=".env.local"):
    """Minimal KEY=VALUE loader so OPEN_AI_API_KEY in .env.local is picked up."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# --- model call ---------------------------------------------------------------

def call_model(cfg, system, user, temperature, max_tokens, timeout):
    """Dispatch to the model's provider. Returns (text, error)."""
    if cfg.get("provider") == "ollama":
        return _call_ollama(cfg, system, user, temperature, max_tokens, timeout)
    return _call_openai(cfg, system, user, temperature, max_tokens, timeout)


def _call_ollama(cfg, system, user, temperature, max_tokens, timeout):
    """Native Ollama /api/chat with think toggle + context/predict control."""
    url = cfg["base_url"].rstrip("/") + "/api/chat"
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": cfg.get("think", False),
        "options": {
            "temperature": temperature,
            "num_ctx": cfg.get("num_ctx", 8192),
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (payload.get("message", {}).get("content") or ""), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _call_openai(cfg, system, user, temperature, max_tokens, timeout):
    """POST an OpenAI-compatible chat completion. Returns (text, error)."""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        cfg.get("token_param", "max_tokens"): max_tokens,
    }
    if cfg.get("supports_temperature", True):
        body["temperature"] = temperature
    body.update(cfg.get("extra_body", {}))
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key_env = cfg.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            return None, f"missing {key_env}"
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    # This network intercepts TLS (verified handshake fails with a non-critical CA
    # basic-constraints error), so use an unverified context. Safe here: user's own
    # machine + key + corporate proxy. Override with cfg["verify_ssl"]=True.
    ctx = ssl.create_default_context() if cfg.get("verify_ssl") else ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        msg = payload["choices"][0]["message"]
        return (msg.get("content") or ""), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:  # noqa: BLE001 - report and move on
        return None, f"{type(e).__name__}: {e}"


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    """Pull the Python program out of a model response.

    Strips qwen-style <think> blocks, prefers a fenced block containing
    generate_crossword, then any fenced block, then a bare def."""
    if not text:
        return None
    text = _THINK.sub("", text)
    blocks = _FENCE.findall(text)
    if blocks:
        for b in blocks:
            if "def generate_crossword" in b:
                return b.strip()
        return max(blocks, key=len).strip()
    if "def generate_crossword" in text:
        # no fence: take from the first import/def to the end
        idx = [text.find(t) for t in ("import ", "from ", "def generate_crossword") if t in text]
        start = min(i for i in idx if i >= 0) if any(i >= 0 for i in idx) else 0
        return text[start:].strip()
    return None


# --- eval set -----------------------------------------------------------------

def load_eval_specs(path, max_specs=None):
    """Load held-out eval records: (spec_id, size, system, user, effective_spec).
    Balances across sizes when max_specs caps the set."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        eff = (d.get("meta") or {}).get("effective_spec") or {}
        rows.append({
            "spec_id": eff.get("spec_id"),
            "size": eff.get("size"),
            "system": d["messages"][0]["content"],
            "user": d["messages"][1]["content"],
            "effective_spec": eff,
        })
    if max_specs and len(rows) > max_specs:
        by_size = {}
        for r in rows:
            by_size.setdefault(r["size"], []).append(r)
        sizes = sorted(by_size)
        picked, i = [], 0
        while len(picked) < max_specs and any(by_size.values()):
            s = sizes[i % len(sizes)]
            if by_size[s]:
                picked.append(by_size[s].pop(0))
            i += 1
        rows = picked
    return rows


# --- scoring ------------------------------------------------------------------

def build_palette():
    edu = build_clean_education_source()
    word_source = {"theme": edu["targets"], "fill": edu["fill_words"]}
    return {
        "word_source": word_source,
        "vocab_set": edu["clean_set"],
        "scores": edu["scores"],
        "targets": edu["targets"],
    }


def make_spec(eff, targets):
    return Spec(
        size=int(eff.get("size", 7)),
        topic_words=tuple(targets),
        require_symmetry=eff.get("require_symmetry", False),
        min_word_len=eff.get("min_word_len", 3),
        time_budget_s=eff.get("time_budget_s", 5.0),
        density_target=eff.get("density_target", 0.72),
    )


def score_code(code, eff, palette):
    """Sandbox-run + score a model's program. Returns a flat metrics dict."""
    spec = make_spec(eff, palette["targets"])
    out = evaluate_code(
        code, spec,
        word_source=palette["word_source"],
        scores=palette["scores"],
        n_draws=1,
        vocab_set=palette["vocab_set"],
        quality_penalty=False,
        in_process=False,  # UNTRUSTED model output -> subprocess sandbox
    )
    m = out["metrics"]
    budget = spec.time_budget_s
    valid = (m.get("valid", 0) or 0) >= 0.999
    filler = m.get("filler_fraction") or 0.0
    inv = (m.get("invalid_crossing_frac") or 0.0) + (m.get("invalid_entry_frac") or 0.0)
    runtime = m.get("runtime_s") or 0.0
    within = valid and filler <= 0.30 and inv == 0.0 and runtime <= budget
    n_entries = m.get("n_entries") or 0.0
    return {
        "ran": 1,
        "valid": 1 if valid else 0,
        "within_spec": 1 if within else 0,
        "coverage": m.get("coverage") or 0.0,
        "crossings": m.get("crossings") or 0.0,
        "n_entries": n_entries,
        "filler_fraction": filler,
        "n_filler": round(filler * n_entries, 2),
        "invalid_crossing_frac": m.get("invalid_crossing_frac") or 0.0,
        "invalid_entry_frac": m.get("invalid_entry_frac") or 0.0,
        "runtime_s": runtime,
        "reasons": out["artifacts"].get("best_draw", {}).get("reasons"),
    }


ZERO = {"ran": 0, "valid": 0, "within_spec": 0, "coverage": 0.0, "crossings": 0.0,
        "n_entries": 0.0, "filler_fraction": 0.0, "n_filler": 0.0,
        "invalid_crossing_frac": 0.0, "invalid_entry_frac": 0.0, "runtime_s": 0.0}


# --- per-model run ------------------------------------------------------------

def run_one(name, cfg, spec_rows, palette, samples, temperature, max_tokens, timeout, log):
    """Generate + score every (spec x sample). Returns list of per-sample records."""
    tasks = []
    for r in spec_rows:
        for s in range(samples):
            tasks.append((r, s))

    def work(item):
        r, s = item
        t0 = time.time()
        text, err = call_model(cfg, r["system"], r["user"], temperature, max_tokens, timeout)
        rec = {"spec_id": r["spec_id"], "size": r["size"], "sample": s,
               "model": name, "user": r["user"], "gen_error": err,
               "gen_s": round(time.time() - t0, 1)}
        code = extract_code(text) if text else None
        rec["parsed"] = 1 if code else 0
        rec["code"] = code
        if not code:
            rec.update(dict(ZERO))
            rec["reasons"] = ["no code emitted" if err is None else err]
            return rec
        try:
            rec.update(score_code(code, r["effective_spec"], palette))
        except Exception as e:  # noqa: BLE001
            rec.update(dict(ZERO))
            rec["reasons"] = [f"score error: {type(e).__name__}: {e}"]
        return rec

    records = []
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 3)) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            # per-generation live update: the PROMPT given, then the EVAL it earned
            mark = "PASS" if rec["within_spec"] else ("valid" if rec["valid"] else "FAIL")
            why = ""
            if not rec["within_spec"]:
                rs = rec.get("reasons") or []
                why = f"  <- {rs[0]}" if rs else ""
            log(f'  [{name} {done}/{len(tasks)}] {rec["spec_id"]} {rec["size"]}x{rec["size"]}'
                f'  prompt: "{rec["user"]}"  ({rec["gen_s"]}s)')
            log(f'      {mark}: valid={rec["valid"]} within_spec={rec["within_spec"]} '
                f'coverage={rec["coverage"]:.2f} connections={int(rec["crossings"])} '
                f'entries={int(rec["n_entries"])} filler={rec["filler_fraction"]*100:.0f}% '
                f'badword={rec["invalid_entry_frac"]*100:.0f}% rt={rec["runtime_s"]:.2f}s{why}')
    return records


def aggregate(name, model_id, records):
    n = len(records)
    parsed = [r for r in records if r["parsed"]]
    ran = [r for r in records if r["ran"]]

    def mean(rows, k):
        return round(sum(r[k] for r in rows) / len(rows), 4) if rows else 0.0

    # pass@k: per (spec_id) is any sample valid?
    by_spec = {}
    for r in records:
        by_spec.setdefault(r["spec_id"], []).append(r)
    pass_at_k = round(sum(1 for v in by_spec.values() if any(x["valid"] for x in v))
                      / len(by_spec), 4) if by_spec else 0.0
    return {
        "model": name,
        "model_id": model_id,
        "n_samples": n,
        "n_prompts": len(by_spec),
        "parse_rate": round(len(parsed) / n, 4) if n else 0.0,
        "valid_rate": round(sum(r["valid"] for r in records) / n, 4) if n else 0.0,
        "within_spec_rate": round(sum(r["within_spec"] for r in records) / n, 4) if n else 0.0,
        "pass_at_k": pass_at_k,
        "mean_coverage": mean(ran, "coverage"),
        "mean_crossings": mean(ran, "crossings"),
        "mean_n_entries": mean(ran, "n_entries"),
        "mean_filler_fraction": mean(ran, "filler_fraction"),
        "mean_n_filler": mean(ran, "n_filler"),
        "mean_invalid_entry_frac": mean(ran, "invalid_entry_frac"),
        "mean_invalid_crossing_frac": mean(ran, "invalid_crossing_frac"),
        "mean_runtime_s": mean(ran, "runtime_s"),
    }


def print_table(aggs):
    cols = [
        ("model", "model", 16, "s"),
        ("n", "n_samples", 5, "d"),
        ("parse%", "parse_rate", 7, "pct"),
        ("valid%", "valid_rate", 7, "pct"),
        ("within%", "within_spec_rate", 8, "pct"),
        ("pass@k", "pass_at_k", 7, "pct"),
        ("cover", "mean_coverage", 7, "f2"),
        ("cross", "mean_crossings", 6, "f1"),
        ("entries", "mean_n_entries", 8, "f1"),
        ("filler%", "mean_filler_fraction", 8, "pct"),
        ("badword%", "mean_invalid_entry_frac", 9, "pct"),
        ("rt(s)", "mean_runtime_s", 7, "f2"),
    ]
    head = "".join(f"{h:>{w}}" if h != "model" else f"{h:<{w}}" for h, _, w, _ in cols)
    print("\n" + head)
    print("-" * len(head))
    for a in aggs:
        cells = []
        for h, key, w, fmt in cols:
            v = a.get(key, 0)
            if fmt == "pct":
                cells.append(f"{v*100:>{w}.1f}")
            elif fmt == "f2":
                cells.append(f"{v:>{w}.2f}")
            elif fmt == "f1":
                cells.append(f"{v:>{w}.1f}")
            elif fmt == "d":
                cells.append(f"{v:>{w}d}")
            else:
                cells.append(f"{str(v):<{w}}")
        print("".join(cells))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Base-vs-tuned crossword-generator eval")
    ap.add_argument("--eval-file", default="runs/bulk/dataset/eval.jsonl")
    ap.add_argument("--models", nargs="+", default=["qwen3-4b-base", "gpt-5.5"],
                    help="model names from the registry, or path to a JSON registry override")
    ap.add_argument("--models-config", default=None, help="JSON file: {name: {base_url,...}}")
    ap.add_argument("--max-specs", type=int, default=None, help="cap eval specs (balanced by size)")
    ap.add_argument("--samples", type=int, default=1, help="completions per spec")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="fallback per-model concurrency if not set in registry")
    ap.add_argument("--parallel-models", action=argparse.BooleanOptionalAction, default=True,
                    help="run all models concurrently (Ollama + OpenAI don't contend)")
    ap.add_argument("--out", default=None, help="output JSON (default runs/eval/eval_<ts>.json)")
    args = ap.parse_args(argv)

    load_env()
    registry = dict(DEFAULT_MODELS)
    if args.models_config and os.path.exists(args.models_config):
        registry.update(json.load(open(args.models_config, encoding="utf-8")))

    spec_rows = load_eval_specs(args.eval_file, args.max_specs)
    print(f"eval file: {args.eval_file}")
    print(f"eval specs: {len(spec_rows)} "
          f"(sizes: {sorted({r['size'] for r in spec_rows})}) x {args.samples} sample(s) "
          f"= {len(spec_rows)*args.samples} generations/model")
    print("building clean educational palette (word_source)...")
    palette = build_palette()
    print(f"palette: {len(palette['vocab_set'])} vocab-crossword words, "
          f"{len(palette['targets'])} SAT targets\n")

    def log(m):
        print(m, flush=True)

    valid_models = []
    for name in args.models:
        cfg = dict(registry.get(name, {}))
        if not cfg:
            print(f"!! unknown model '{name}' (not in registry); skipping")
            continue
        cfg.setdefault("concurrency", args.concurrency)
        valid_models.append((name, cfg))
        print(f"== {name}  ({cfg.get('model')} @ {cfg.get('base_url')}) "
              f"conc={cfg['concurrency']} ==", flush=True)

    def run_model(name, cfg):
        t0 = time.time()
        recs = run_one(name, cfg, spec_rows, palette, args.samples,
                       args.temperature, args.max_tokens, args.timeout, log)
        dt = time.time() - t0
        a = aggregate(name, cfg.get("model"), recs)
        a["wall_s"] = round(dt, 1)
        return name, recs, a

    all_records, agg_by_name = {}, {}
    if args.parallel_models and len(valid_models) > 1:
        # local Ollama and the OpenAI API don't contend -> run both at once,
        # each with its own internal concurrency. Wall time = max, not sum.
        with ThreadPoolExecutor(max_workers=len(valid_models)) as mex:
            futs = [mex.submit(run_model, n, c) for n, c in valid_models]
            for f in as_completed(futs):
                name, recs, a = f.result()
                all_records[name] = recs
                agg_by_name[name] = a
                print(f"   [done] {name}: {a['wall_s']:.0f}s, valid {a['valid_rate']*100:.1f}%, "
                      f"within-spec {a['within_spec_rate']*100:.1f}%\n", flush=True)
    else:
        for name, cfg in valid_models:
            name, recs, a = run_model(name, cfg)
            all_records[name] = recs
            agg_by_name[name] = a
            print(f"   [done] {name}: {a['wall_s']:.0f}s, valid {a['valid_rate']*100:.1f}%, "
                  f"within-spec {a['within_spec_rate']*100:.1f}%\n", flush=True)

    aggs = [agg_by_name[n] for n, _ in valid_models if n in agg_by_name]
    print_table(aggs)

    out = args.out or f"runs/eval/eval_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "aggregates": aggs,
                   "records": all_records}, fh, indent=2)
    print(f"wrote {out}")
    return aggs


if __name__ == "__main__":
    main()
