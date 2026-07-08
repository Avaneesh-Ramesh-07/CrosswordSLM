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

 (MD, "## 2. Check the GPU\nThe eval harness is pure-Python stdlib and Ollama self-installs in the "
      "serve cell, so there's nothing to pip-install. Just confirm a GPU is attached — if not, "
      "**Runtime → Change runtime type → T4 GPU**."),
 (CO, '!nvidia-smi -L || echo "NO GPU — set Runtime > Change runtime type > T4 GPU, then rerun"'),

 (MD, "## 3. Pull the CrossWordBench data (gated)\n"
      "The dataset is gated, so you need a HuggingFace **read** token whose account has "
      "**accepted the terms** at huggingface.co/datasets/HINT-lab/CrossWordBench. The token is "
      "read via getpass (not stored in the notebook). Pulls the `english` config (7x7 + 14x14) "
      "as compact JSONL into `data/crosswordbench/`."),
 (CO, 'import os, getpass\n'
      'os.environ["HF_TOKEN"] = getpass.getpass("HF read token (gated access accepted): ")\n'
      '!python bench/pull_crosswordbench.py --config english\n'
      '!ls -la data/crosswordbench/'),

 (MD, "## 4. Serve Qwen3-4B via Ollama\n"
      "vLLM on Colab often breaks on CUDA-version drift (missing `libcudart`); Ollama bundles its own "
      "CUDA runners, so it just works. This installs the Linux x86-64 build (asset looked up "
      "dynamically so a version bump won't break it), starts the server on :11434, and pulls the "
      "model. Default tag is Q4 (~2.6GB); set `MODEL = \"qwen3:4b-fp16\"` (~8GB) for full precision."),
 (CO, r'''# Serve Qwen3-4B via Ollama (dynamic asset lookup; zstd tarball)
import subprocess, time, requests

# 1) find the current linux x86-64 build (asset name/format changes across releases)
rel = requests.get("https://api.github.com/repos/ollama/ollama/releases/latest", timeout=30).json()
assets = {a["name"]: a for a in rel.get("assets", [])}
name = "ollama-linux-amd64.tar.zst"
assert name in assets, f"{name} not in release {rel.get('tag_name')}: {sorted(assets)}"
url, mb = assets[name]["browser_download_url"], assets[name]["size"] // 1024 // 1024
print(f"downloading ollama {rel['tag_name']} :: {name} ({mb}MB)…", flush=True)

# 2) download + extract to /usr  (-f fails visibly on HTTP error, -L follows redirects)
subprocess.run(f"curl -fSL '{url}' -o /tmp/ollama.tar.zst", shell=True, check=True)
subprocess.run("command -v zstd >/dev/null || apt-get -qq install -y zstd", shell=True)
subprocess.run("tar -I zstd -xf /tmp/ollama.tar.zst -C /usr", shell=True, check=True)
print("ollama:", subprocess.run(["/usr/bin/ollama", "--version"],
      capture_output=True, text=True).stdout.strip() or "installed", flush=True)

# 3) start the server in the background
srv = subprocess.Popen(["/usr/bin/ollama", "serve"],
                       stdout=open("ollama.log", "w"), stderr=subprocess.STDOUT)
print(f"launched ollama pid={srv.pid}", flush=True)

def _tail(p="ollama.log", n=6):
    try:
        with open(p) as fh: return "".join(fh.readlines()[-n:]).rstrip()
    except FileNotFoundError: return "(no log yet)"

# 4) wait for the API (crash-aware)
up, start = False, time.time()
while time.time() - start < 120:
    el = int(time.time() - start)
    rc = srv.poll()
    if rc is not None:
        print(f"[{el}s] ollama serve EXITED rc={rc}. Log:\n{_tail(n=20)}", flush=True); break
    try:
        if requests.get("http://localhost:11434/api/tags", timeout=2).ok:
            up = True; print(f"[{el}s] ollama UP on :11434", flush=True); break
    except Exception:
        pass
    if el % 15 < 3: print(f"[{el:4d}s] starting… | {_tail(n=1)}", flush=True)
    time.sleep(3)

# 5) pull the model (progress streams to the cell)
MODEL = "qwen3:4b"              # Q4 (~2.6GB). fp16: "qwen3:4b-fp16" (~8GB)
if up:
    print(f"pulling {MODEL} (first time only)…", flush=True)
    subprocess.run(["/usr/bin/ollama", "pull", MODEL], check=True)
    print(f"{MODEL} ready — serving OpenAI-compatible API at http://localhost:11434/v1", flush=True)'''),

 (MD, "## 5. Baselines / anchors (no model needed)\n"
      "Every run reports BOTH success rates:\n"
      "- **strict** (NYT rules: every cell checked both ways + symmetry) — references score ~0 "
      "(they're loose auto-fills), so this is a hard target.\n"
      "- **relaxed** (CrossWordBench-style: unchecked cells allowed, but every cell in a real entry, "
      "no conflicts, connected) — references score ~1.0, so `success` is meaningful.\n\n"
      "`reference` scores each puzzle's OWN grid (crossings ceiling). `seed:reference_v1` runs a "
      "hand-written CSP generator as a 'model' — a floor showing how hard a valid fill is from only "
      "the ~12 given words."),
 (CO, "!python bench/run_crosswordbench.py --model reference --config english\n"
      "!python bench/run_crosswordbench.py --model seed:reference_v1 --config english --limit 20"),

 (MD, "## 6. Smoke test: Qwen3-4B on 20 puzzles\n"
      "Confirms the endpoint round-trip and response parsing before the full run. "
      "`--mode program` = model writes a `generate_crossword` program we run in the sandbox "
      "(matches your trained SLM's interface). Use `--mode direct` to have it emit the layout JSON itself."),
 (CO, "!python bench/run_crosswordbench.py --model endpoint --mode program \\\n"
      "    --base-url http://localhost:11434/v1 --model-name qwen3:4b \\\n"
      "    --config english --split 7x7 --limit 20"),

 (MD, "## 7. Full eval + save per-puzzle results\n"
      "Two runs over all 200 english puzzles — one per interface. Each generates once and reports "
      "**both** strict and relaxed success (plus crossings / coverage / black-delta). `--mode program` "
      "matches your trained SLM's interface; `--mode direct` has the model emit the layout JSON itself. "
      "Per-puzzle rows go to `runs/eval/*.jsonl` for comparison against the tuned SLM. Strict success "
      "may be ~0 — the relaxed success + crossings + coverage columns are where the signal is."),
 (CO, 'import os\n'
      'os.makedirs("runs/eval", exist_ok=True)\n'
      'EP = "--base-url http://localhost:11434/v1 --model-name qwen3:4b --config english"\n'
      '!python bench/run_crosswordbench.py --model endpoint --mode program $EP \\\n'
      '    --out runs/eval/qwen3_4b_program.jsonl\n'
      '!python bench/run_crosswordbench.py --model endpoint --mode direct $EP \\\n'
      '    --out runs/eval/qwen3_4b_direct.jsonl'),

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
