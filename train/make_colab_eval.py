"""Generate train/colab_eval.ipynb (the CrossWordBench base-model eval run).

Companion to make_colab.py. This notebook serves Qwen3-4B via vLLM and runs the
CrossWordBench-derived eval (bench/) INSIDE the Colab session against
localhost:8000 -- no tunnel. It reports the binary success rate (valid config)
plus the nuance metrics (crossings, coverage, black-square delta), with the
reference grid and a seed generator as baselines/anchors.

Run:  python train/make_colab_eval.py
"""

import json
import os

MD = "markdown"
CO = "code"

cells = [
 (MD, "# CrossWordBench Eval — Qwen3-4B baseline\n\n"
      "Serve **Qwen3-4B** with vLLM and run the CrossWordBench-derived eval in the same "
      "session (against `localhost:8000` — **no tunnel**). Each puzzle's word set + size + "
      "black-square count become a SPEC; the model produces a crossword configuration; we "
      "score **success** (valid config) plus **crossings**, **coverage**, and **black-square "
      "delta**.\n\n"
      "**Runtime:** GPU (T4 is enough for a 4B model). **Order:** run top to bottom."),

 (MD, "## 1. Get the code\nClone the repo (set your URL) or upload the project folder and "
      "set `PROJECT_DIR`. The `bench/`, `harness/`, `pipeline/`, and `seeds/` dirs must be present."),
 (CO, 'REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed — upload the folder instead"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "bench")), "Set PROJECT_DIR to the repo root"\n'
      '%cd $PROJECT_DIR'),

 (MD, "## 2. Install dependencies\nThe eval harness is pure-Python stdlib; we only need vLLM "
      "(to serve the model) and requests (to poll it)."),
 (CO, "!pip -q install vllm requests"),

 (MD, "## 3. Pull the CrossWordBench data (gated)\n"
      "The dataset is gated, so you need a HuggingFace **read** token whose account has "
      "**accepted the terms** at huggingface.co/datasets/HINT-lab/CrossWordBench. The token is "
      "read via getpass (not stored in the notebook). Pulls the `english` config (7x7 + 14x14) "
      "as compact JSONL into `data/crosswordbench/`."),
 (CO, 'import os, getpass\n'
      'os.environ["HF_TOKEN"] = getpass.getpass("HF read token (gated access accepted): ")\n'
      '!python bench/pull_crosswordbench.py --config english\n'
      '!ls -la data/crosswordbench/'),

 (MD, "## 4. Serve Qwen3-4B via vLLM\n"
      "T4: keep `--dtype half`. A100/L4: `bfloat16` and a larger `--max-model-len` are fine."),
 (CO,
      'import subprocess, sys, time, requests\n'
      '\n'
      'MODEL = "Qwen/Qwen3-4B"\n'
      'LOG = "vllm.log"\n'
      'server = subprocess.Popen(\n'
      '    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",\n'
      '     "--model", MODEL, "--dtype", "half", "--max-model-len", "8192",\n'
      '     "--gpu-memory-utilization", "0.90", "--port", "8000"],\n'
      '    stdout=open(LOG, "w"), stderr=subprocess.STDOUT)\n'
      'print(f"launched vLLM pid={server.pid} for {MODEL}; first run downloads ~8GB…", flush=True)\n'
      '\n'
      'def _tail(n=8):\n'
      '    try:\n'
      '        with open(LOG) as fh: return "".join(fh.readlines()[-n:]).rstrip()\n'
      '    except FileNotFoundError: return "(no log yet)"\n'
      '\n'
      'start, DEADLINE, up, i = time.time(), 1500, False, 0\n'
      'while time.time() - start < DEADLINE:\n'
      '    el = int(time.time() - start)\n'
      '    rc = server.poll()\n'
      '    if rc is not None:                       # crashed -> stop now, show why\n'
      '        print(f"\\n[{el}s] vLLM EXITED rc={rc}. Last log:\\n", flush=True)\n'
      '        print(_tail(40), flush=True); break\n'
      '    try:\n'
      '        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:\n'
      '            up = True; print(f"\\n[{el}s] vLLM UP — {MODEL} ready on :8000", flush=True); break\n'
      '    except Exception:\n'
      '        pass\n'
      '    if i % 3 == 0:                           # heartbeat + latest log line ~every 15s\n'
      '        print(f"[{el:4d}s] loading… | {_tail(1)}", flush=True)\n'
      '    i += 1; time.sleep(5)\n'
      'if not up and server.poll() is None:\n'
      '    print(f"\\n[{int(time.time()-start)}s] TIMEOUT — still not serving.", flush=True)\n'
      'if not up:\n'
      '    print("\\n===== tail vllm.log =====\\n" + _tail(40), flush=True)'),

 (MD, "## 5. Baselines / anchors (no model needed)\n"
      "`reference` scores each puzzle's OWN grid. Two validity modes:\n"
      "- **strict** (NYT rules: every cell checked both ways + symmetry) — references score ~0 "
      "(they're loose auto-fills), so this is a hard target.\n"
      "- **relaxed** (`--relaxed`, CrossWordBench-style: unchecked cells allowed, but every cell in "
      "a real entry, no conflicts, connected) — references score ~1.0, so `success` is meaningful.\n\n"
      "`seed:reference_v1` runs a hand-written CSP generator as a 'model' — a floor showing how hard "
      "a valid fill is from only the ~12 given words."),
 (CO, "!python bench/run_crosswordbench.py --model reference --config english            # strict\n"
      "!python bench/run_crosswordbench.py --model reference --config english --relaxed  # relaxed (≈1.0)\n"
      "!python bench/run_crosswordbench.py --model seed:reference_v1 --config english --limit 20 --relaxed"),

 (MD, "## 6. Smoke test: Qwen3-4B on 20 puzzles\n"
      "Confirms the endpoint round-trip and response parsing before the full run. "
      "`--mode program` = model writes a `generate_crossword` program we run in the sandbox "
      "(matches your trained SLM's interface). Use `--mode direct` to have it emit the layout JSON itself."),
 (CO, "!python bench/run_crosswordbench.py --model endpoint --mode program \\\n"
      "    --base-url http://localhost:8000/v1 --model-name Qwen/Qwen3-4B \\\n"
      "    --config english --split 7x7 --limit 20"),

 (MD, "## 7. Full eval + save per-puzzle results\n"
      "Runs all 200 english puzzles. `--mode program` (matches your trained SLM's interface) is "
      "reported under both **strict** and **relaxed** validity; `--mode direct` (model emits the "
      "layout JSON) under relaxed. Per-puzzle rows go to `runs/eval/*.jsonl` for later analysis and "
      "for comparison against the tuned SLM. Success may be ~0 under strict — the crossings / "
      "coverage / black-delta columns are where the signal is."),
 (CO, 'import os\n'
      'os.makedirs("runs/eval", exist_ok=True)\n'
      'EP = "--base-url http://localhost:8000/v1 --model-name Qwen/Qwen3-4B --config english"\n'
      '!python bench/run_crosswordbench.py --model endpoint --mode program $EP \\\n'
      '    --out runs/eval/qwen3_4b_program_strict.jsonl\n'
      '!python bench/run_crosswordbench.py --model endpoint --mode program $EP --relaxed \\\n'
      '    --out runs/eval/qwen3_4b_program_relaxed.jsonl\n'
      '!python bench/run_crosswordbench.py --model endpoint --mode direct $EP --relaxed \\\n'
      '    --out runs/eval/qwen3_4b_direct_relaxed.jsonl'),

 (MD, "## 8. Save results to Drive"),
 (CO, 'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      '!mkdir -p /content/drive/MyDrive/slm_runs/eval\n'
      '!cp -r runs/eval/* /content/drive/MyDrive/slm_runs/eval/ 2>/dev/null || true\n'
      'print("saved runs/eval to Drive")'),

 (MD, "## Next\n"
      "These are the **base-model** numbers. After QLoRA training, re-run cell 7 pointing "
      "`--model-name` at the tuned checkpoint (served the same way) to get the base-vs-tuned "
      "delta on identical held-out puzzles."),
]


def build():
    nb_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == CO:
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_eval.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    return out


if __name__ == "__main__":
    path = build()
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    print(f"wrote {path}")
    print(f"cells: {len(nb['cells'])} "
          f"({sum(c['cell_type']=='code' for c in nb['cells'])} code, "
          f"{sum(c['cell_type']=='markdown' for c in nb['cells'])} md)")
