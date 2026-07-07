"""Generate train/colab_openevolve.ipynb (the OpenEvolve data-generation run).

Authoring the notebook via a builder keeps the .ipynb valid JSON. Run:
    python train/make_colab.py
Only covers the OpenEvolve step (serve LLM -> evolve -> harvest -> dataset);
QLoRA training + base-vs-tuned eval are a separate notebook.
"""

import json
import os

MD = "markdown"
CO = "code"

cells = [
 (MD, "# Crossword-Generator Dataset — OpenEvolve Run\n\n"
      "This notebook runs **only the data-generation step**: serve Qwen3-4B as the search model, "
      "evolve crossword-generator programs with OpenEvolve against our deterministic scorer, and "
      "harvest the trace into a SOAR-style `(spec -> program)` dataset. QLoRA training + base-vs-tuned "
      "eval come in a separate notebook.\n\n"
      "**Runtime:** GPU (T4 works; A100 faster). **Order:** run cells top to bottom."),

 (MD, "## 1. Get the code\nClone the repo (set your URL) or upload the project folder and set `PROJECT_DIR`."),
 (CO, 'REPO_URL = "https://github.com/YOUR_USER/YOUR_REPO.git"   # <-- set this, or upload the folder\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "pipeline")), "Set PROJECT_DIR to the repo root"\n'
      '%cd $PROJECT_DIR'),

 (MD, "## 2. Install dependencies\nOur pipeline is pure-Python stdlib; the word lists ship in `data/`."),
 (CO, "!pip -q install openevolve vllm requests"),

 (MD, "## 3. Confirm the OpenEvolve API (catch version drift)\n"
      "We built `pipeline/oe_evaluator.py` and `pipeline/run_openevolve.py` against the documented API. "
      "This cell checks the evaluator import and the CLI entry point. If either differs, update "
      "`oe_evaluator.py`'s `EvaluationResult` import and `run_openevolve.py`'s `_run_openevolve_cli`."),
 (CO, 'import openevolve\n'
      'print("openevolve version:", getattr(openevolve, "__version__", "?"))\n'
      'try:\n'
      '    from openevolve.evaluation_result import EvaluationResult\n'
      '    print("EvaluationResult import OK")\n'
      'except Exception as e:\n'
      '    print("EvaluationResult import DIFFERS ->", e)\n'
      'print("top-level names:", [n for n in dir(openevolve) if not n.startswith("_")])\n'
      '!python -m openevolve.cli --help 2>&1 | head -n 15 || echo "no openevolve.cli — check the CLI entry point"'),

 (MD, "## 4. Serve Qwen3-4B via vLLM (the search model)\n"
      "T4: keep `--dtype half`. A100/L4: you may use `bfloat16` and a larger `--max-model-len`."),
 (CO, 'import subprocess, sys, time, requests\n'
      'MODEL = "Qwen/Qwen3-4B"\n'
      'log = open("vllm.log", "w")\n'
      'server = subprocess.Popen(\n'
      '    [sys.executable, "-m", "vllm.entrypoints.openai.api_server",\n'
      '     "--model", MODEL, "--dtype", "half", "--max-model-len", "8192",\n'
      '     "--gpu-memory-utilization", "0.90", "--port", "8000"],\n'
      '    stdout=log, stderr=subprocess.STDOUT)\n'
      'up = False\n'
      'for _ in range(120):\n'
      '    try:\n'
      '        if requests.get("http://localhost:8000/v1/models", timeout=2).ok:\n'
      '            up = True; break\n'
      '    except Exception:\n'
      '        pass\n'
      '    time.sleep(10)\n'
      'print("vLLM up" if up else "NOT up — see vllm.log below")\n'
      '!tail -n 20 vllm.log'),

 (MD, "## 5. Run OpenEvolve across the pilot specs\n"
      "The driver seeds OpenEvolve with **three distinct generator families** — `reference_v1` "
      "(greedy + backtracking), `csp_ac3` (CSP + AC-3 propagation), and `beam_search` (greedy beam) — "
      "running one evolution per (seed x spec) and merging every evaluated candidate into a single "
      "harvest. Multiple seeds give evolution diverse starting points and a more varied dataset.\n\n"
      "Pilot uses sizes **7,9** (all seeds bootstrap a valid grid there). Each candidate is scored by our "
      "sandbox+scorer and appended to `runs/pilot/harvest.jsonl`; the driver then builds "
      "`runs/pilot/dataset/{train,dev,test}.jsonl`. Start small, then scale `--n-specs` / `--iterations`."),
 (CO, "!python pipeline/run_openevolve.py \\\n"
      "    --seeds reference_v1,csp_ac3,beam_search \\\n"
      "    --sizes 7,9 --n-specs 8 --iterations 60 \\\n"
      "    --model Qwen/Qwen3-4B --api-base http://localhost:8000/v1 \\\n"
      "    --out runs/pilot"),

 (MD, "## 6. Inspect the harvested dataset"),
 (CO, 'import json, os\n'
      'h = [json.loads(l) for l in open("runs/pilot/harvest.jsonl")]\n'
      'print("candidates harvested:", len(h))\n'
      'for split in ("train", "dev", "test"):\n'
      '    p = f"runs/pilot/dataset/{split}.jsonl"\n'
      '    n = sum(1 for _ in open(p)) if os.path.exists(p) else 0\n'
      '    print(f"  {split}: {n}")\n'
      'tp = "runs/pilot/dataset/train.jsonl"\n'
      'if os.path.exists(tp) and os.path.getsize(tp):\n'
      '    ex = json.loads(open(tp).readline())\n'
      '    print("\\n--- sample SPEC ---\\n", ex["messages"][1]["content"][:300])\n'
      '    print("\\n--- sample PROGRAM (head) ---\\n", ex["messages"][2]["content"][:400])'),

 (MD, "## 7. Save outputs to Drive"),
 (CO, 'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      '!mkdir -p /content/drive/MyDrive/slm_runs\n'
      '!cp -r runs/pilot /content/drive/MyDrive/slm_runs/\n'
      'print("saved runs/pilot to Drive")'),

 (MD, "## Next\n"
      "With `dataset/train.jsonl` in hand, the **training notebook** QLoRA-fine-tunes Qwen3-4B on it and "
      "runs the base-vs-tuned eval (reusing our sandbox+scorer), then exports a GGUF for local inference."),
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_openevolve.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    return out


if __name__ == "__main__":
    path = build()
    # validate it round-trips as JSON and has the expected cells
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    print(f"wrote {path}")
    print(f"cells: {len(nb['cells'])} "
          f"({sum(c['cell_type']=='code' for c in nb['cells'])} code, "
          f"{sum(c['cell_type']=='markdown' for c in nb['cells'])} md)")
