"""Generate train/colab_eval_tuned.ipynb — EVAL 3 for the tuned crossword SLM.

Runs EVAL 3 (held-out eval.jsonl, BARE deployment prompt) on the merged fine-tuned
model, scored the SAME way as Claude's EVAL 3 (pipeline.eval_opus_fleet.score_one on the
clean English palette + real-dictionary check). This is the base-vs-tuned headline:
Claude Opus scored 0/100 on these identical bare prompts (GAP_ANALYSIS EVAL 3).

Why no separate "as-is" run (Claude EVAL 3 had one): Claude didn't conform to our
generate_crossword contract, so we ran its programs on their own interface to rule out an
API-mismatch artifact. The tuned model is trained to emit a conforming
generate_crossword(topic, word_source, size) FUNCTION, so calling it and scoring the
returned layout (what score_one does) already IS its own-terms test.

    python train/make_colab_eval3.py
"""

import json
import os

MD, CO = "markdown", "code"

REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# EVAL 3 — tuned Qwen3-4B on held-out `eval.jsonl` (bare deployment prompt)\n\n"
      "The base-vs-tuned **headline**. Each of the held-out `eval.jsonl` specs is fed to the "
      "tuned model as the **bare** system+user prompt it was trained on (no contract in the "
      "prompt — the contract lives in the weights). Every emitted `generate_crossword` program "
      "is scored through the **same** sandbox + scorer + real-dictionary check as Claude's "
      "EVAL 3 (`pipeline.eval_opus_fleet.score_one`).\n\n"
      "**Comparison:** unaugmented Claude Opus scored **0/100** on these identical bare prompts "
      "(GAP_ANALYSIS EVAL 3).\n\n"
      "> **No separate \"as-is\" run.** Claude's EVAL 3 also ran each program on its own "
      "interface to rule out an API-mismatch artifact — needed only because Claude didn't "
      "conform to our contract. The tuned model emits a conforming `generate_crossword` "
      "*function*, so calling it and scoring the layout (below) already is the own-terms test. "
      "(Cell 8 optionally dumps the raw programs for manual inspection.)\n\n"
      "**Runtime:** GPU. L4 (24 GB) / A100 (40 GB) recommended; T4 works but generation is slower. "
      "**Order:** run top to bottom."),

 (MD, "## 1. Get the code\n"
      "Clone the repo (set your URL) or upload the folder and set `PROJECT_DIR`. The clone "
      "already contains everything the eval needs: `data/sft/eval.jsonl`, the word lists in "
      "`data/wordlists/` (incl. `words_alpha.txt`), and the `pipeline/` + `harness/` scoring "
      "code.\n\n"
      "> Push your latest local changes first — the eval reflects whatever is committed here."),
 (CO, 'REPO_URL = "%s"\n'
      'import os\n'
      '!git clone -q $REPO_URL slm || echo "clone skipped/failed — upload the folder instead"\n'
      'PROJECT_DIR = "/content/slm"   # adjust if you uploaded elsewhere\n'
      'assert os.path.isdir(os.path.join(PROJECT_DIR, "pipeline")), "Set PROJECT_DIR to the repo root"\n'
      '%%cd $PROJECT_DIR' % REPO_URL),

 (MD, "## 2. GPU + install deps\n"
      "Colab already ships torch. We add `transformers`/`accelerate` (pinned to the training "
      "snapshot) to load the merged model, and `wordfreq` (the English palette intersects the "
      "wordfreq top-N). No bitsandbytes — the merged model loads in 16-bit directly."),
 (CO, 'import torch\n'
      'assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > GPU (L4/A100 recommended)"\n'
      'print("GPU:", torch.cuda.get_device_name(0))\n'
      '!pip install -q "transformers==4.53.*" "accelerate==1.8.*" wordfreq'),

 (MD, "> **Expected pip warning — safe to ignore.** Colab's pre-installed `gradio` wants a "
      "newer `huggingface-hub` than `transformers 4.53` pins. `gradio` is unused here; do **not** "
      "upgrade `huggingface-hub` (it would break `transformers`)."),

 (MD, "## 3. Point to the merged tuned model (on Drive)\n"
      "Mount Drive and set `MODEL_DIR` to your merged model folder "
      "(`…/qwen3-4b-crossword-qlora-merged`) — the full standalone model, not the adapter."),
 (CO, 'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      '# EDIT this to your merged-model path on Drive:\n'
      'MODEL_DIR = "/content/drive/MyDrive/qwen3-4b-crossword-qlora-merged"\n'
      'import os\n'
      'assert os.path.isdir(MODEL_DIR), f"MODEL_DIR not found: {MODEL_DIR}"\n'
      'assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), \\\n'
      '    "not a full model dir (need config.json + model-*.safetensors, i.e. the -merged folder, not the adapter)"\n'
      'print("model dir OK:", MODEL_DIR)'),

 (MD, "## 4. Load the tuned model"),
 (CO, 'import torch\n'
      'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
      'tok = AutoTokenizer.from_pretrained(MODEL_DIR)\n'
      'if tok.pad_token_id is None:\n'
      '    tok.pad_token = tok.eos_token\n'
      'tok.padding_side = "left"   # left-pad so batched generation aligns at the prompt end\n'
      'model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto")\n'
      'model.eval()\n'
      'print("loaded:", model.config.model_type, "| dtype", next(model.parameters()).dtype, "| device", model.device)'),

 (MD, "## 5. Generation settings + batched helper\n"
      "`GEN_TEMP = 1.0` matches Claude's EVAL 3 (temperature 1.0). Set it to `0.0` for greedy / "
      "deterministic decoding (the tuned model's single best output). `MAX_NEW_TOKENS` is generous; "
      "the tuned programs are compact — raise it only if you see truncated code."),
 (CO, 'GEN_TEMP       = 1.0     # match Claude EVAL 3; use 0.0 for greedy/deterministic\n'
      'MAX_NEW_TOKENS = 3072   # tuned programs are compact; raise if any get truncated\n'
      'BATCH          = 8      # lower if you OOM on a small GPU\n\n'
      'import torch\n'
      '@torch.no_grad()\n'
      'def generate_batch(pairs):\n'
      '    """pairs: list of (system, user) -> list of completion strings (assistant turn only)."""\n'
      '    outs = []\n'
      '    for i in range(0, len(pairs), BATCH):\n'
      '        chunk = pairs[i:i + BATCH]\n'
      '        texts = [tok.apply_chat_template(\n'
      '                    [{"role": "system", "content": s}, {"role": "user", "content": u}],\n'
      '                    tokenize=False, add_generation_prompt=True)\n'
      '                 for (s, u) in chunk]\n'
      '        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,\n'
      '                  max_length=2048).to(model.device)\n'
      '        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,\n'
      '                             do_sample=GEN_TEMP > 0, temperature=max(GEN_TEMP, 1e-5),\n'
      '                             top_p=0.95, pad_token_id=tok.pad_token_id)\n'
      '        new = gen[:, enc["input_ids"].shape[1]:]\n'
      '        outs.extend(tok.batch_decode(new, skip_special_tokens=True))\n'
      '        print(f"  generated {min(i + BATCH, len(pairs))}/{len(pairs)}", flush=True)\n'
      '    return outs'),

 (MD, "## 6. EVAL 3 — generate on bare eval.jsonl prompts, score through the harness\n"
      "Identical to Claude's EVAL 3: sizes 7/9/11/15, 25 prompts per size (n=100, drawn from "
      "`eval.jsonl` with the same seed), scored by `score_one` on the clean English palette with "
      "a real-dictionary check on every entry."),
 (CO, 'import os, json, time\n'
      '# these modules read ANTHROPIC_* via os.environ.get; set dummies so nothing complains\n'
      'os.environ.setdefault("ANTHROPIC_BASE_URL", ""); os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "")\n'
      'from pipeline.eval_opus_evalset import load_prompts\n'
      'from pipeline.eval_opus_fleet import score_one, agg, table\n'
      'from pipeline.eval_selfmodel import english_palette\n'
      'from pipeline.eval_harness import extract_code\n\n'
      'SIZES = [7, 9, 11, 15]; PER_SIZE = 25\n'
      'prompts = load_prompts("data/sft/eval.jsonl", SIZES, PER_SIZE)   # (system, user, size), BARE\n'
      'print(f"{len(prompts)} bare prompts")\n'
      'print(f"  example -> system={prompts[0][0]!r}\\n             user={prompts[0][1]!r}")\n'
      'pal = english_palette(max(SIZES))\n\n'
      'print("generating...", flush=True)\n'
      'comps = generate_batch([(s, u) for (s, u, sz) in prompts])\n\n'
      'print("scoring...", flush=True)\n'
      'rows = []\n'
      'for (s, u, sz), txt in zip(prompts, comps):\n'
      '    code = extract_code(txt)\n'
      '    if not code:\n'
      '        rec = {"valid": 0, "fully": 0, "within": 0, "dict_frac": 0.0, "coverage": 0.0,\n'
      '               "crossings": 0, "entries": 0, "filler": 0.0, "parsed": 0}\n'
      '    else:\n'
      '        rec = score_one(code, pal, sz, "vocabulary"); rec["parsed"] = 1\n'
      '    rec["size"] = sz; rows.append(rec)\n\n'
      'parse_rate = sum(r["parsed"] for r in rows) / len(rows)\n'
      'print(f"\\nparse rate (emitted a code block): {parse_rate*100:.0f}%")\n'
      'ov = table(f"TUNED Qwen3-4B on eval.jsonl BARE prompts (n={len(rows)}, temp={GEN_TEMP})", rows, SIZES)'),

 (MD, "## 7. Save results + base-vs-tuned comparison"),
 (CO, 'import os, json, time\n'
      'os.makedirs("runs/eval", exist_ok=True)\n'
      'out = f"runs/eval/tuned_evalset_{int(time.time())}.json"\n'
      'summary = {"model": "qwen3-4b-crossword-qlora-merged", "condition": "bare eval.jsonl prompts",\n'
      '           "n": len(rows), "parse_rate": parse_rate, "gen_temp": GEN_TEMP, "overall": ov,\n'
      '           "by_size": {s: agg([r for r in rows if r["size"] == s]) for s in SIZES}}\n'
      'json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)\n'
      'print("wrote", out)\n'
      '!mkdir -p /content/drive/MyDrive/slm_runs/eval && cp "$out" /content/drive/MyDrive/slm_runs/eval/ 2>/dev/null || true\n\n'
      'print("\\n===== EVAL 3 (bare eval.jsonl, harness-scored) — base vs tuned =====")\n'
      'print(f"  Claude Opus 4.8 : valid  0%   fullyOK  0%   within  0%    [GAP_ANALYSIS EVAL 3, n=100]")\n'
      'print(f"  Tuned Qwen3-4B  : valid {ov[\'valid\']*100:3.0f}%   fullyOK {ov[\'fully\']*100:3.0f}%   "\n'
      '      f"within {ov[\'within\']*100:3.0f}%   (n={len(rows)}, temp={GEN_TEMP})")'),

 (MD, "## 8. (Optional) dump the raw programs for inspection\n"
      "Parallels Claude's saved as-is programs. Writes each emitted program to "
      "`runs/eval/tuned_progs/` so you can eyeball the actual generators the model produced."),
 (CO, 'import os\n'
      'os.makedirs("runs/eval/tuned_progs", exist_ok=True)\n'
      'for k, ((s, u, sz), txt) in enumerate(zip(prompts, comps)):\n'
      '    code = extract_code(txt) or txt\n'
      '    open(f"runs/eval/tuned_progs/prog_{k:03d}_s{sz}.py", "w", encoding="utf-8").write(code)\n'
      'print("saved", len(prompts), "programs under runs/eval/tuned_progs/")'),

 (MD, "## Next\n"
      "If the tuned model clears a meaningful bar here (vs Claude's 0%), record the numbers in "
      "`GAP_ANALYSIS.md` as the tuned column of EVAL 3. To also run EVAL 1/2 (clean-room / Spanish), "
      "reuse `english_palette`/`spanish_palette` + `score_one` with the fleet prompt the same way."),
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_eval_tuned.ipynb")
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
